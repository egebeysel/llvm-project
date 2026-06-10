// RUN: mlir-opt %s -transform-interpreter -canonicalize -cse -split-input-file --verify-diagnostics | FileCheck %s

// Perfect scalable tiling - scalable tile sizes equal scalable inner
// tiles. Outer sizes of the tiled unpack should be 1's.

// CHECK-LABEL: func.func @perfect_CKkc_to_KC_scalable
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:            tensor<1x1x?x?xf32> -> tensor<?x?xf32>
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             tensor.insert_slice %[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @perfect_CKkc_to_KC_scalable(%source: tensor<32x4x?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c2_vscale = arith.muli %c2, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %0 = linalg.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
      inner_tiles = [%c2_vscale, %c4_vscale] into %dest
      : tensor<32x4x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [[2], [4]]
          {inner_tile_alignments = array<i64: 2, 2>}
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// Aligned scalable tiling - scalable tile sizes ([16] = 16*vscale, [8] =
// 8*vscale) that are integer multiples of the scalable inner tiles (2x 8*vscale
// and 2x 4*vscale), so tiling is aligned and the tiled unpack needs no trailing
// remainder slice.

// CHECK-LABEL: func.func @NCnc_to_NC_scalable_aligned
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:            tensor<?x?x?x?xf32> -> tensor<?x?xf32>
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             tensor.insert_slice %[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @NCnc_to_NC_scalable_aligned(%source: tensor<4x8x?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %0 = linalg.unpack %source inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %dest
      : tensor<4x8x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [[16], [8]]
          {inner_tile_alignments = array<i64: 1, 1>}
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// Unaligned scalable tiling - static tile sizes not aligned to scalable
// inner tiles.

// CHECK-LABEL: func.func @NCnc_to_NC_scalable_unaligned
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:            tensor<?x?x?x?xf32> -> tensor<?x?xf32>
// CHECK:             %[[EXTRACT:.*]] = tensor.extract_slice %[[UNPACK]]
// CHECK:             tensor.insert_slice %[[EXTRACT]]
// CHECK:         return %[[RES]]
func.func @NCnc_to_NC_scalable_unaligned(%source: tensor<4x8x?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %0 = linalg.unpack %source inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %dest
      : tensor<4x8x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [7, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// Producer fusion - linalg.unpack with scalable inner tiles fused as a producer
// into an elementwise consumer (linalg.exp) that is tiled with scalable tile
// sizes that are an integer multiple of the inner tiles (16*vscale of 8*vscale,
// 8*vscale of 4*vscale). The consumer is tiled via transform.structured.fuse
// with handles to the payload `arith.muli` tile-size values (no scalable literal
// syntax), and the Multiple hint lets the unpack fuse without re-deriving the
// relationship from the scalable IR.

// CHECK-LABEL: func.func @unpack_elemwise_scalable_multiple
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// Multiple (not Equal): outer dims stay dynamic (ceilDiv), not collapsed to 1.
// CHECK-SAME:            tensor<?x?x?x?xf32> -> tensor<?x?xf32>
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             linalg.exp ins(%[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @unpack_elemwise_scalable_multiple(%arg0: tensor<4x8x?x?xf32>, %arg1: tensor<?x?xf32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %t0 = arith.muli %c16, %vscale : index
  %t1 = arith.muli %c8, %vscale : index
  %0 = tensor.empty(%arg2, %arg3) : tensor<?x?xf32>
  %1 = linalg.unpack %arg0 inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %0
      : tensor<4x8x?x?xf32> -> tensor<?x?xf32>
  %2 = linalg.exp ins(%1: tensor<?x?xf32>)
                       outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %exp = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mulis = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %i0, %i1, %t0h, %t1h = transform.split_handle %mulis : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %tiled, %loops:2 = transform.structured.fuse %exp tile_sizes [%t0h, %t1h] interchange [0, 1]
      {inner_tile_alignments = array<i64: 1, 1>}
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Producer fusion - same as above but the consumer is tiled with scalable tile
// sizes equal to the unpack inner tiles (8*vscale, 4*vscale), supplied as
// distinct payload SSA values. The Equal hint collapses the fused unpack's outer
// dims to 1 even though the tile-size and inner-tile SSA values are not provably
// equal (no scalable literal syntax / no out-of-tree fuse support needed).

// CHECK-LABEL: func.func @unpack_elemwise_scalable_equal
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:            tensor<1x1x?x?xf32> -> tensor<?x?xf32>
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             linalg.exp ins(%[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @unpack_elemwise_scalable_equal(%arg0: tensor<4x8x?x?xf32>, %arg1: tensor<?x?xf32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %t0 = arith.muli %c8, %vscale : index
  %t1 = arith.muli %c4, %vscale : index
  %0 = tensor.empty(%arg2, %arg3) : tensor<?x?xf32>
  %1 = linalg.unpack %arg0 inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %0
      : tensor<4x8x?x?xf32> -> tensor<?x?xf32>
  %2 = linalg.exp ins(%1: tensor<?x?xf32>)
                       outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %exp = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mulis = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %i0, %i1, %t0h, %t1h = transform.split_handle %mulis : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %tiled, %loops:2 = transform.structured.fuse %exp tile_sizes [%t0h, %t1h] interchange [0, 1]
      {inner_tile_alignments = array<i64: 2, 2>}
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Producer fusion with a TRANSPOSING consumer. The consumer reads the unpack
// result transposed ((d0, d1) -> (d1, d0)), so consumer dim d0 maps to unpack
// dest dim 1 and d1 to dest dim 0. inner_tile_alignments is a caller-asserted
// hint indexed in the unpack's dest-dim order, so the caller must arrange it for
// the transpose. The consumer is tiled [d0 = 4*vscale, d1 = 16*vscale], so the
// fused unpack sees dim0 tile = 16*vscale (2x its 8*vscale inner tile -> Multiple)
// and dim1 tile = 4*vscale (= its 4*vscale inner tile -> Equal). The hint
// array<i64: 1, 2> ([Multiple, Equal]) is arranged accordingly, so dim0 keeps a
// non-unit outer extent and only dim1 collapses to 1 -- i.e. the transpose is
// handled correctly because the hint accounts for it.

// CHECK-LABEL: func.func @unpack_transposed_consumer_scalable
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[SRC:.*]] = tensor.extract_slice
// CHECK-SAME:            tensor<2x4x?x?xf32> to tensor<?x1x?x?xf32>
// CHECK:             %[[UNPACK:.*]] = linalg.unpack %[[SRC]]
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             linalg.generic
// CHECK:         return %[[RES]]
func.func @unpack_transposed_consumer_scalable(%arg0: tensor<2x4x?x?xf32>, %out: tensor<?x?xf32>, %d0: index, %d1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %t0 = arith.muli %c4, %vscale : index
  %t1 = arith.muli %c16, %vscale : index
  %0 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %0
      : tensor<2x4x?x?xf32> -> tensor<?x?xf32>
  // Consumer reads %unpack transposed.
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                                        affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
       ins(%unpack : tensor<?x?xf32>) outs(%out : tensor<?x?xf32>) {
    ^bb0(%in: f32, %o: f32):
      linalg.yield %in : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %mulis = transform.structured.match ops{["arith.muli"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %i0, %i1, %h0, %h1 = transform.split_handle %mulis
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)
    // Hint indexed in the unpack's dest-dim order: dim0 Multiple, dim1 Equal.
    %tiled, %loops:2 = transform.structured.fuse %gen tile_sizes [%h0, %h1]
        {inner_tile_alignments = array<i64: 1, 2>}
        : (!transform.any_op, !transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion - linalg.unpack with scalable inner tiles fused as a
// consumer into an scf.for loop. The loop step on the inner tile dimension
// equals the unpack inner tile size (8*vscale), so fusion succeeds.

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @fuse_scalable_unpack_consumer
// CHECK-SAME:      %[[ARG0:.+]]: tensor<32x?xf32>, %[[ARG1:.+]]: tensor<32x?xf32>, %[[ARG2:.+]]: tensor<32x?xf32>
//      CHECK:    %[[VSCALE:.*]] = vector.vscale
//      CHECK:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %{{.*}} : index
//      CHECK:    %[[RES:.*]]:2 = scf.for {{.*}} step %[[C8_VSCALE]]
// CHECK-SAME:        iter_args(%{{.*}} = %[[ARG2]], %{{.*}} = %{{.*}})
//      CHECK:      linalg.generic
//      CHECK:      %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:          inner_tiles = [%[[C8_VSCALE]]]
//      CHECK:      scf.yield {{.*}}, %{{.*}} :
//      CHECK:    return %[[RES]]#1
func.func @fuse_scalable_unpack_consumer(
    %arg0: tensor<32x?xf32>, %arg1: tensor<32x?xf32>,
    %arg2: tensor<32x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %dim1 = tensor.dim %arg2, %c1 : tensor<32x?xf32>

  %0 = scf.for %iv = %c0 to %dim1 step %c8_vscale iter_args(%out = %arg2) -> (tensor<32x?xf32>) {
    %extracted = tensor.extract_slice %out[0, %iv] [32, %c8_vscale] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<32x?xf32>, tensor<32x?xf32>)
        outs(%extracted : tensor<32x?xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<32x?xf32>
    %inserted = tensor.insert_slice %computed into %out[0, %iv] [32, %c8_vscale] [1, 1]
        : tensor<32x?xf32> into tensor<32x?xf32>
    scf.yield %inserted : tensor<32x?xf32>
  }

  %output = tensor.empty(%dim1) : tensor<?xf32>
  %unpack = linalg.unpack %0 outer_dims_perm = [0]
      inner_dims_pos = [0] inner_tiles = [%c8_vscale]
      into %output : tensor<32x?xf32> -> tensor<?xf32>
  return %unpack : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %unpack into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion (negative) - linalg.unpack with scalable inner tiles and no
// alignment hint. The loop step (4*vscale) and the unpack inner tile (8*vscale)
// are distinct SSA values, so without a hint equality is not statically provable
// and fusion falls through to failure.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_scalable_unpack_consumer_mismatch(
    %arg0: tensor<32x?xf32>, %arg1: tensor<32x?xf32>,
    %arg2: tensor<32x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index
  %dim1 = tensor.dim %arg2, %c1 : tensor<32x?xf32>

  %0 = scf.for %iv = %c0 to %dim1 step %c4_vscale iter_args(%out = %arg2) -> (tensor<32x?xf32>) {
    %extracted = tensor.extract_slice %out[0, %iv] [32, %c4_vscale] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<32x?xf32>, tensor<32x?xf32>)
        outs(%extracted : tensor<32x?xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<32x?xf32>
    %inserted = tensor.insert_slice %computed into %out[0, %iv] [32, %c4_vscale] [1, 1]
        : tensor<32x?xf32> into tensor<32x?xf32>
    scf.yield %inserted : tensor<32x?xf32>
  }

  %output = tensor.empty(%dim1) : tensor<?xf32>
  // expected-error @below {{'linalg.unpack' op failed to fuse consumer of slice}}
  %unpack = linalg.unpack %0 outer_dims_perm = [0]
      inner_dims_pos = [0] inner_tiles = [%c8_vscale]
      into %output : tensor<32x?xf32> -> tensor<?xf32>
  return %unpack : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %unpack into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion with an Equal hint and a non-suffix inner_dims_pos. The
// unpack packs dest dim 0 (inner_dims_pos = [0]), so its inner tile maps to a
// leading dest dim rather than a trailing one. The loop step (8*vscale) and the
// unpack inner tile (%tile) are distinct SSA values, so equality is not
// statically provable; the caller asserts it on dest dim 0 via
// inner_tile_alignments = array<i64: 2, 0> ([Equal, Unknown]). Fusion must read
// the hint at dest dim 0, not at the trailing dim, for this to succeed.

#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @fuse_unpack_consumer_nonsuffix_equal_hint
//      CHECK:    %[[RES:.*]]:2 = scf.for
//      CHECK:      linalg.generic
//      CHECK:      %[[UNPACK:.*]] = linalg.unpack
//  CHECK-NOT:      tensor.extract_slice %[[UNPACK]]
//      CHECK:      scf.yield
//      CHECK:    return %[[RES]]#1
func.func @fuse_unpack_consumer_nonsuffix_equal_hint(
    %arg0: tensor<4x16x?xf32>, %arg1: tensor<4x16x?xf32>,
    %arg2: tensor<4x16x?xf32>, %tile: index) -> tensor<?x16xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %dim2 = tensor.dim %arg2, %c2 : tensor<4x16x?xf32>

  %0 = scf.for %iv = %c0 to %dim2 step %c8_vscale iter_args(%out = %arg2) -> (tensor<4x16x?xf32>) {
    %ext_out = tensor.extract_slice %out[0, 0, %iv] [4, 16, %c8_vscale] [1, 1, 1]
        : tensor<4x16x?xf32> to tensor<4x16x?xf32>
    %ext_a = tensor.extract_slice %arg0[0, 0, %iv] [4, 16, %c8_vscale] [1, 1, 1]
        : tensor<4x16x?xf32> to tensor<4x16x?xf32>
    %ext_b = tensor.extract_slice %arg1[0, 0, %iv] [4, 16, %c8_vscale] [1, 1, 1]
        : tensor<4x16x?xf32> to tensor<4x16x?xf32>
    %computed = linalg.generic {
        indexing_maps = [#map3, #map3, #map3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<4x16x?xf32>, tensor<4x16x?xf32>)
        outs(%ext_out : tensor<4x16x?xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<4x16x?xf32>
    %inserted = tensor.insert_slice %computed into %out[0, 0, %iv] [4, 16, %c8_vscale] [1, 1, 1]
        : tensor<4x16x?xf32> into tensor<4x16x?xf32>
    scf.yield %inserted : tensor<4x16x?xf32>
  }

  %d0 = arith.muli %c4, %tile : index
  %output = tensor.empty(%d0) : tensor<?x16xf32>
  %unpack = linalg.unpack %0 inner_dims_pos = [0] inner_tiles = [%tile]
      into %output : tensor<4x16x?xf32> -> tensor<?x16xf32>
  return %unpack : tensor<?x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %unpack into (%loop) {inner_tile_alignments = array<i64: 2, 0>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion (negative) - same non-suffix unpack as above, but without an
// inner_tile_alignments hint. The loop step and the unpack inner tile are
// distinct SSA values, so equality is not statically provable and, absent the
// caller's assertion, the consumer cannot be fused.

#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @fuse_unpack_consumer_nonsuffix_no_hint(
    %arg0: tensor<4x16x?xf32>, %arg1: tensor<4x16x?xf32>,
    %arg2: tensor<4x16x?xf32>, %tile: index) -> tensor<?x16xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %dim2 = tensor.dim %arg2, %c2 : tensor<4x16x?xf32>

  %0 = scf.for %iv = %c0 to %dim2 step %c8_vscale iter_args(%out = %arg2) -> (tensor<4x16x?xf32>) {
    %ext_out = tensor.extract_slice %out[0, 0, %iv] [4, 16, %c8_vscale] [1, 1, 1]
        : tensor<4x16x?xf32> to tensor<4x16x?xf32>
    %ext_a = tensor.extract_slice %arg0[0, 0, %iv] [4, 16, %c8_vscale] [1, 1, 1]
        : tensor<4x16x?xf32> to tensor<4x16x?xf32>
    %ext_b = tensor.extract_slice %arg1[0, 0, %iv] [4, 16, %c8_vscale] [1, 1, 1]
        : tensor<4x16x?xf32> to tensor<4x16x?xf32>
    %computed = linalg.generic {
        indexing_maps = [#map3, #map3, #map3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<4x16x?xf32>, tensor<4x16x?xf32>)
        outs(%ext_out : tensor<4x16x?xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<4x16x?xf32>
    %inserted = tensor.insert_slice %computed into %out[0, 0, %iv] [4, 16, %c8_vscale] [1, 1, 1]
        : tensor<4x16x?xf32> into tensor<4x16x?xf32>
    scf.yield %inserted : tensor<4x16x?xf32>
  }

  %d0 = arith.muli %c4, %tile : index
  %output = tensor.empty(%d0) : tensor<?x16xf32>
  // expected-error @below {{'linalg.unpack' op failed to fuse consumer of slice}}
  %unpack = linalg.unpack %0 inner_dims_pos = [0] inner_tiles = [%tile]
      into %output : tensor<4x16x?xf32> -> tensor<?x16xf32>
  return %unpack : tensor<?x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %unpack into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Standalone tiling (static contradiction) - the loop tile sizes (7, 5) are
// statically unequal to the inner tiles (8, 4), so an `Equal` hint must be
// ignored and the unpack remainder slice must survive.

// CHECK-LABEL: func.func @unpack_static_contradicting_equal_hint
// CHECK:         %[[UNPACK:.*]] = linalg.unpack
// CHECK:         tensor.extract_slice %[[UNPACK]]
func.func @unpack_static_contradicting_equal_hint(%source: tensor<?x?x8x4xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.unpack %source inner_dims_pos = [0, 1] inner_tiles = [8, 4]
      into %dest : tensor<?x?x8x4xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [7, 5]
        {inner_tile_alignments = array<i64: 2, 2>}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion (negative, static contradiction) - the loop step (4) and the
// unpack inner tile (8) are statically unequal, so an `Equal` hint must be
// ignored and the consumer cannot be fused.

#map_static = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @fuse_unpack_consumer_static_contradicting_equal_hint(
    %arg0: tensor<4x16x8xf32>, %arg1: tensor<4x16x8xf32>,
    %arg2: tensor<4x16x8xf32>) -> tensor<32x16xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %0 = scf.for %iv = %c0 to %c8 step %c4 iter_args(%out = %arg2) -> (tensor<4x16x8xf32>) {
    %ext_out = tensor.extract_slice %out[0, 0, %iv] [4, 16, 4] [1, 1, 1]
        : tensor<4x16x8xf32> to tensor<4x16x4xf32>
    %ext_a = tensor.extract_slice %arg0[0, 0, %iv] [4, 16, 4] [1, 1, 1]
        : tensor<4x16x8xf32> to tensor<4x16x4xf32>
    %ext_b = tensor.extract_slice %arg1[0, 0, %iv] [4, 16, 4] [1, 1, 1]
        : tensor<4x16x8xf32> to tensor<4x16x4xf32>
    %computed = linalg.generic {
        indexing_maps = [#map_static, #map_static, #map_static],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<4x16x4xf32>, tensor<4x16x4xf32>)
        outs(%ext_out : tensor<4x16x4xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<4x16x4xf32>
    %inserted = tensor.insert_slice %computed into %out[0, 0, %iv] [4, 16, 4] [1, 1, 1]
        : tensor<4x16x4xf32> into tensor<4x16x8xf32>
    scf.yield %inserted : tensor<4x16x8xf32>
  }

  %output = tensor.empty() : tensor<32x16xf32>
  // expected-error @below {{'linalg.unpack' op failed to fuse consumer of slice}}
  %unpack = linalg.unpack %0 inner_dims_pos = [0] inner_tiles = [8]
      into %output : tensor<4x16x8xf32> -> tensor<32x16xf32>
  return %unpack : tensor<32x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %unpack into (%loop) {inner_tile_alignments = array<i64: 2, 0>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion (negative) - the loop step (16*vscale) is a non-unit multiple
// of the unpack inner tile (8*vscale), so the inner dim is genuinely tiled. Only
// `Equal` is meaningful for an unpack consumer's inner dim; a `Multiple` hint is
// not sufficient, so fusion must still fail.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_scalable_unpack_consumer_multiple_hint_fails(
    %arg0: tensor<32x?xf32>, %arg1: tensor<32x?xf32>,
    %arg2: tensor<32x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c16_vscale = arith.muli %c16, %vscale : index
  %dim1 = tensor.dim %arg2, %c1 : tensor<32x?xf32>

  %0 = scf.for %iv = %c0 to %dim1 step %c16_vscale iter_args(%out = %arg2) -> (tensor<32x?xf32>) {
    %extracted = tensor.extract_slice %out[0, %iv] [32, %c16_vscale] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<32x?xf32>, tensor<32x?xf32>)
        outs(%extracted : tensor<32x?xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<32x?xf32>
    %inserted = tensor.insert_slice %computed into %out[0, %iv] [32, %c16_vscale] [1, 1]
        : tensor<32x?xf32> into tensor<32x?xf32>
    scf.yield %inserted : tensor<32x?xf32>
  }

  %output = tensor.empty(%dim1) : tensor<?xf32>
  // expected-error @below {{'linalg.unpack' op failed to fuse consumer of slice}}
  %unpack = linalg.unpack %0 outer_dims_perm = [0]
      inner_dims_pos = [0] inner_tiles = [%c8_vscale]
      into %output : tensor<32x?xf32> -> tensor<?xf32>
  return %unpack : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %unpack into (%loop) {inner_tile_alignments = array<i64: 1>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Standalone tiling with a non-identity `interchange` and an asymmetric hint.
// The hint is indexed in iteration-domain order (d0 = Equal, d1 = Multiple), not
// in interchanged order: so the fused unpack collapses its d0 outer dim to 1 and
// keeps d1 dynamic, while `interchange = [1, 0]` makes the d1 loop the outermost.

// CHECK-LABEL: func.func @unpack_interchange_hint(
// CHECK-SAME:      %[[SRC:.*]]: tensor<?x?x?x?xf32>, %[[D0:.*]]: index, %[[D1:.*]]: index
// CHECK:         scf.for {{.*}} to %[[D1]]
// CHECK:           scf.for {{.*}} to %[[D0]]
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:            tensor<1x?x?x?xf32> -> tensor<?x?xf32>
func.func @unpack_interchange_hint(%source: tensor<?x?x?x?xf32>, %d0: index, %d1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %dest = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.unpack %source outer_dims_perm = [0, 1] inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %dest
      : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // d0 tile (8*vscale) = inner (8*vscale) -> Equal; d1 tile (8*vscale) is 2x
    // the inner (4*vscale) -> Multiple. Hint in iteration-domain order.
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [[8], [8]]
        interchange = [1, 0] {inner_tile_alignments = array<i64: 2, 1>}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Standalone tiling with a static loop tile and a scalable inner tile under an
// `Equal` hint. The tile (8/4) and the inner tile (8*vscale/4*vscale) cannot be
// related statically, so the hint is consulted and trusted as a caller
// assertion: the unpack is aligned (no trailing remainder slice) and the outer
// dims collapse to a static 1. (The hint is never re-checked; an incorrect
// assertion would silently mistile -- by design.)

// CHECK-LABEL: func.func @unpack_static_tile_scalable_inner_hint
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:            tensor<1x1x?x?xf32> -> tensor<?x?xf32>
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             tensor.insert_slice %[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @unpack_static_tile_scalable_inner_hint(
    %source: tensor<4x8x?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %0 = linalg.unpack %source inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %dest
      : tensor<4x8x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Static tile [8, 4], scalable inner [8*vscale, 4*vscale] -> not statically
    // decidable; Equal hint -> aligned (ceilDiv), but bounded tile -> no collapse.
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [8, 4]
        {inner_tile_alignments = array<i64: 2, 2>}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The hint array is positional and unchecked against the iteration-domain rank:
// extra trailing entries are silently ignored (and, symmetrically, a shorter
// array leaves trailing dims `Unknown`). Here a 3-entry hint is given for a
// rank-2 unpack; the extra entry is dropped and the two `Equal` entries still
// collapse the outer dims to 1.

// CHECK-LABEL: func.func @unpack_oversized_hint_extra_ignored
// CHECK:         %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:        tensor<1x1x?x?xf32> -> tensor<?x?xf32>
func.func @unpack_oversized_hint_extra_ignored(
    %source: tensor<32x4x?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c2_vscale = arith.muli %c2, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %0 = linalg.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
      inner_tiles = [%c2_vscale, %c4_vscale] into %dest
      : tensor<32x4x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // 3 entries for a rank-2 iteration domain: the trailing entry is ignored.
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [[2], [4]]
        {inner_tile_alignments = array<i64: 2, 2, 2>}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Fusing a scalable `linalg.unpack` producer into a containing `scf.forall` via
// `transform.structured.fuse_into_containing_op`. With the `inner_tile_alignments`
// hint the producer is tiled in its aligned form: the source slice collapses to a
// single inner tile (`tensor<1x1x?x?xf32>`) and the unpacked result feeds the
// consumer directly, with no recovery `tensor.extract_slice`.

// CHECK-LABEL: func.func @fuse_unpack_into_containing_aligned
//       CHECK:   scf.forall
//       CHECK:     %[[SRC:.+]] = tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [1, 1, %{{.*}}, %{{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<1x1x?x?xf32>
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack %[[SRC]]
//  CHECK-SAME:         : tensor<1x1x?x?xf32> -> tensor<?x?xf32>
//   CHECK-NOT:     tensor.extract_slice %[[UNPACK]]
//       CHECK:     linalg.exp ins(%[[UNPACK]]
func.func @fuse_unpack_into_containing_aligned(
    %src: tensor<?x?x?x?xf32>, %unpack_empty: tensor<?x?xf32>,
    %out: tensor<?x?xf32>, %ub0: index, %ub1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %unpack_empty
      : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %res = scf.forall (%i, %j) = (0, 0) to (%ub0, %ub1) step (%c8_vscale, %c4_vscale)
      shared_outs(%o = %out) -> (tensor<?x?xf32>) {
    %slice = tensor.extract_slice %unpack[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %oslice = tensor.extract_slice %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %0 = linalg.exp ins(%slice : tensor<?x?xf32>) outs(%oslice : tensor<?x?xf32>) -> tensor<?x?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
    }
  }
  return %res : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, %newc = transform.structured.fuse_into_containing_op %0 into %1
        {inner_tile_alignments = array<i64: 2, 2>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Same fusion WITHOUT the hint: the producer falls back to the general (unaligned)
// tiling, which over-allocates the unpacked tile (`tensor<?x?x?x?xf32>` source) and
// recovers the needed slice with a trailing `tensor.extract_slice` on the result.

// CHECK-LABEL: func.func @fuse_unpack_into_containing_unaligned
//       CHECK:   scf.forall
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack
//  CHECK-SAME:         : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
//       CHECK:     tensor.extract_slice %[[UNPACK]]
func.func @fuse_unpack_into_containing_unaligned(
    %src: tensor<?x?x?x?xf32>, %unpack_empty: tensor<?x?xf32>,
    %out: tensor<?x?xf32>, %ub0: index, %ub1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %unpack_empty
      : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %res = scf.forall (%i, %j) = (0, 0) to (%ub0, %ub1) step (%c8_vscale, %c4_vscale)
      shared_outs(%o = %out) -> (tensor<?x?xf32>) {
    %slice = tensor.extract_slice %unpack[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %oslice = tensor.extract_slice %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %0 = linalg.exp ins(%slice : tensor<?x?xf32>) outs(%oslice : tensor<?x?xf32>) -> tensor<?x?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
    }
  }
  return %res : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, %newc = transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The hint also reaches the block-argument fusion path: when the producer is the
// `scf.forall` init (used through the block argument rather than via a direct
// extract use), `inner_tile_alignments` still yields the aligned tiling.

// CHECK-LABEL: func.func @fuse_unpack_through_block_arg_aligned
//       CHECK:   scf.forall
//       CHECK:     %[[SRC:.+]] = tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [1, 1, %{{.*}}, %{{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<1x1x?x?xf32>
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack %[[SRC]]
//  CHECK-SAME:         : tensor<1x1x?x?xf32> -> tensor<?x?xf32>
//   CHECK-NOT:     tensor.extract_slice %[[UNPACK]]
//       CHECK:     linalg.exp ins(%[[UNPACK]]
func.func @fuse_unpack_through_block_arg_aligned(
    %src: tensor<?x?x?x?xf32>, %unpack_empty: tensor<?x?xf32>,
    %ub0: index, %ub1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %unpack_empty
      : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %res = scf.forall (%i, %j) = (0, 0) to (%ub0, %ub1) step (%c8_vscale, %c4_vscale)
      shared_outs(%o = %unpack) -> (tensor<?x?xf32>) {
    %slice = tensor.extract_slice %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %0 = linalg.exp ins(%slice : tensor<?x?xf32>) outs(%slice : tensor<?x?xf32>) -> tensor<?x?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
    }
  }
  return %res : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, %newc = transform.structured.fuse_into_containing_op %0 into %1
        {inner_tile_alignments = array<i64: 2, 2>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
