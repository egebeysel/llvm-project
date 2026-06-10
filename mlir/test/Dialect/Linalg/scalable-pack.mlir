// RUN: mlir-opt %s -transform-interpreter -canonicalize -cse -split-input-file --verify-diagnostics | FileCheck %s

// Consumer fusion - linalg.pack with scalable inner tiles.
// Producer step (8*vscale) equals the pack inner tile size
// (8*vscale) on the tiled source dimension, so the outer
// dim of the fused pack tile is statically 1.

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @fuse_scalable_pack_consumer_equal
// CHECK-SAME:      %[[ARG0:.+]]: tensor<256x128xf32>, %[[ARG1:.+]]: tensor<256x128xf32>, %[[ARG2:.+]]: tensor<256x128xf32>, %[[DEST:.+]]: tensor<?x?x?x?xf32>
//      CHECK:    %[[C8:.*]] = arith.constant 8 : index
//      CHECK:    %[[VSCALE:.*]] = vector.vscale
//      CHECK:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
//      CHECK:    %[[RES:.*]]:2 = scf.for {{.*}} step %[[C8_VSCALE]]
// CHECK-SAME:        iter_args(%{{.*}} = %[[ARG2]], %{{.*}} = %[[DEST]])
//      CHECK:      %[[GENERIC:.*]] = linalg.generic
//      CHECK:      %[[PACK:.*]] = linalg.pack %[[GENERIC]]
// CHECK-SAME:          inner_tiles = [%[[C8_VSCALE]], %{{.*}}]
// CHECK-SAME:          -> tensor<1x?x?x?xf32>
//      CHECK:      scf.yield {{.*}}, %{{.*}} :
//      CHECK:    return %[[RES]]#1
func.func @fuse_scalable_pack_consumer_equal(
    %arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>,
    %arg2: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index

  %0 = scf.for %iv = %c0 to %c256 step %c8_vscale iter_args(%out = %arg2) -> (tensor<256x128xf32>) {
    %ext_out = tensor.extract_slice %out[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<?x128xf32>, tensor<?x128xf32>)
        outs(%ext_out : tensor<?x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<?x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<?x128xf32> into tensor<256x128xf32>
    scf.yield %inserted : tensor<256x128xf32>
  }

  %pack = linalg.pack %0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %pack into (%loop) {inner_tile_alignments = array<i64: 2, 0>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion with a static producer step (64) and a scalable pack inner
// tile (8*vscale), hinted `Multiple`. This is a caller assertion: 64 is a
// multiple of 8*vscale only for the vscale values where 8*vscale divides 64,
// not universally (the hint is never re-checked). The relationship is not
// statically decidable, so fusion takes the aligned (non-equal) path and the
// outer dim of the fused pack tile is dynamic (`64 ceildiv 8*vscale`).

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @fuse_scalable_pack_consumer_aligned
// CHECK-SAME:      %[[ARG0:.+]]: tensor<256x128xf32>, %[[ARG1:.+]]: tensor<256x128xf32>, %[[ARG2:.+]]: tensor<256x128xf32>, %[[DEST:.+]]: tensor<?x?x?x?xf32>
//      CHECK:    %[[C64:.*]] = arith.constant 64 : index
//      CHECK:    %[[RES:.*]]:2 = scf.for {{.*}} step %[[C64]]
// CHECK-SAME:        iter_args(%{{.*}} = %[[ARG2]], %{{.*}} = %[[DEST]])
//      CHECK:      %[[GENERIC:.*]] = linalg.generic
//      CHECK:      %[[PACK:.*]] = linalg.pack %[[GENERIC]]
// CHECK-SAME:          -> tensor<?x?x?x?xf32>
//      CHECK:      scf.yield {{.*}}, %{{.*}} :
//      CHECK:    return %[[RES]]#1
func.func @fuse_scalable_pack_consumer_aligned(
    %arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>,
    %arg2: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index

  %0 = scf.for %iv = %c0 to %c256 step %c64 iter_args(%out = %arg2) -> (tensor<256x128xf32>) {
    %ext_out = tensor.extract_slice %out[%iv, 0] [64, 128] [1, 1]
        : tensor<256x128xf32> to tensor<64x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [64, 128] [1, 1]
        : tensor<256x128xf32> to tensor<64x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [64, 128] [1, 1]
        : tensor<256x128xf32> to tensor<64x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<64x128xf32>, tensor<64x128xf32>)
        outs(%ext_out : tensor<64x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<64x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [64, 128] [1, 1]
        : tensor<64x128xf32> into tensor<256x128xf32>
    scf.yield %inserted : tensor<256x128xf32>
  }

  %pack = linalg.pack %0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %pack into (%loop) {inner_tile_alignments = array<i64: 1, 0>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion - both producer step and pack inner tile are scalable,
// step (8*vscale) is an integer multiple of the inner tile (4*vscale) but
// not equal to it. Fusion succeeds via the aligned (non-equal) path, so the
// outer dim of the fused pack tile stays dynamic
// (`8*vscale ceildiv 4*vscale`).

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @fuse_scalable_pack_consumer_aligned_scalable
// CHECK-SAME:      %[[ARG0:.+]]: tensor<256x128xf32>, %[[ARG1:.+]]: tensor<256x128xf32>, %[[ARG2:.+]]: tensor<256x128xf32>, %[[DEST:.+]]: tensor<?x?x?x?xf32>
//      CHECK:    %[[C8:.*]] = arith.constant 8 : index
//      CHECK:    %[[VSCALE:.*]] = vector.vscale
//      CHECK:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
//      CHECK:    %[[RES:.*]]:2 = scf.for {{.*}} step %[[C8_VSCALE]]
// CHECK-SAME:        iter_args(%{{.*}} = %[[ARG2]], %{{.*}} = %[[DEST]])
//      CHECK:      %[[GENERIC:.*]] = linalg.generic
//      CHECK:      %[[PACK:.*]] = linalg.pack %[[GENERIC]]
// CHECK-SAME:          -> tensor<?x?x?x?xf32>
//      CHECK:      scf.yield {{.*}}, %{{.*}} :
//      CHECK:    return %[[RES]]#1
func.func @fuse_scalable_pack_consumer_aligned_scalable(
    %arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>,
    %arg2: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index

  %0 = scf.for %iv = %c0 to %c256 step %c8_vscale iter_args(%out = %arg2) -> (tensor<256x128xf32>) {
    %ext_out = tensor.extract_slice %out[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<?x128xf32>, tensor<?x128xf32>)
        outs(%ext_out : tensor<?x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<?x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<?x128xf32> into tensor<256x128xf32>
    scf.yield %inserted : tensor<256x128xf32>
  }

  %pack = linalg.pack %0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [%c4_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %pack into (%loop) {inner_tile_alignments = array<i64: 1, 0>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion (negative): linalg.pack with scalable inner tiles and no
// alignment hint. The loop step (12*vscale) and the inner tile (8*vscale) are
// both dynamic, so without a hint their relationship is not statically decidable
// and fusion falls through to failure.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_scalable_pack_consumer_mismatch(
    %arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>,
    %arg2: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c256 = arith.constant 256 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index
  %c12_vscale = arith.muli %c12, %vscale : index

  %0 = scf.for %iv = %c0 to %c256 step %c12_vscale iter_args(%out = %arg2) -> (tensor<256x128xf32>) {
    %ext_out = tensor.extract_slice %out[%iv, 0] [%c12_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [%c12_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [%c12_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<?x128xf32>, tensor<?x128xf32>)
        outs(%ext_out : tensor<?x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<?x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [%c12_vscale, 128] [1, 1]
        : tensor<?x128xf32> into tensor<256x128xf32>
    scf.yield %inserted : tensor<256x128xf32>
  }

  // expected-error @below {{'linalg.pack' op failed to fuse consumer of slice}}
  %pack = linalg.pack %0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %pack into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion (negative, static contradiction): the loop step (48) and the
// pack inner tile (32) on the tiled dimension are both static and 48 is not a
// multiple of 32. With both sizes static the divisibility check governs and the
// contradicting `Multiple` hint is ignored, so fusion fails.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_pack_consumer_static_contradicting_hint(
    %arg0: tensor<96x128xf32>, %arg1: tensor<96x128xf32>,
    %arg2: tensor<96x128xf32>, %dest: tensor<?x?x32x4xf32>) -> tensor<?x?x32x4xf32> {
  %c0 = arith.constant 0 : index
  %c48 = arith.constant 48 : index
  %c96 = arith.constant 96 : index

  %0 = scf.for %iv = %c0 to %c96 step %c48 iter_args(%out = %arg2) -> (tensor<96x128xf32>) {
    %ext_out = tensor.extract_slice %out[%iv, 0] [48, 128] [1, 1]
        : tensor<96x128xf32> to tensor<48x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [48, 128] [1, 1]
        : tensor<96x128xf32> to tensor<48x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [48, 128] [1, 1]
        : tensor<96x128xf32> to tensor<48x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<48x128xf32>, tensor<48x128xf32>)
        outs(%ext_out : tensor<48x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<48x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [48, 128] [1, 1]
        : tensor<48x128xf32> into tensor<96x128xf32>
    scf.yield %inserted : tensor<96x128xf32>
  }

  // expected-error @below {{'linalg.pack' op failed to fuse consumer of slice}}
  %pack = linalg.pack %0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [32, 4]
      into %dest : tensor<96x128xf32> -> tensor<?x?x32x4xf32>
  return %pack : tensor<?x?x32x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %pack into (%loop) {inner_tile_alignments = array<i64: 1, 0>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion with a transposing outer_dims_perm. The hint is read in source
// (operand) order, so `Equal` on source dim 0 (loop step 8*vscale == inner tile
// 8*vscale) collapses that outer tile to 1; outer_dims_perm = [1, 0] then places
// it at result position 1 (`tensor<?x1x?x?xf32>`), not position 0.

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @fuse_pack_consumer_transposed_outer
//      CHECK:    %[[C8:.*]] = arith.constant 8 : index
//      CHECK:    %[[VSCALE:.*]] = vector.vscale
//      CHECK:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
//      CHECK:    scf.for {{.*}} step %[[C8_VSCALE]]
//      CHECK:      linalg.pack
// CHECK-SAME:          outer_dims_perm = [1, 0]
// CHECK-SAME:          -> tensor<?x1x?x?xf32>
func.func @fuse_pack_consumer_transposed_outer(
    %arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>,
    %arg2: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index

  %0 = scf.for %iv = %c0 to %c256 step %c8_vscale iter_args(%out = %arg2) -> (tensor<256x128xf32>) {
    %ext_out = tensor.extract_slice %out[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<?x128xf32>, tensor<?x128xf32>)
        outs(%ext_out : tensor<?x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<?x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [%c8_vscale, 128] [1, 1]
        : tensor<?x128xf32> into tensor<256x128xf32>
    scf.yield %inserted : tensor<256x128xf32>
  }

  %pack = linalg.pack %0 outer_dims_perm = [1, 0]
      inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %pack into (%loop) {inner_tile_alignments = array<i64: 2, 0>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion where the loop tile is a BOUNDED scalable size: 256 is not a
// static multiple of 8*vscale, so the per-iteration tile is
// affine.min(256 - iv, 8*vscale), whose value-bounds upper bound (256) equals
// the source dim size. The dim must still be treated as tiled and the `Equal`
// hint applied (outer dim collapses to a static 1) -- the bounded upper bound
// must not make the scalable dim look untiled and bypass the hint.

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @fuse_pack_consumer_bounded_scalable_equal
//      CHECK:    %[[PACK:.*]] = linalg.pack
// CHECK-SAME:        -> tensor<1x?x?x?xf32>
func.func @fuse_pack_consumer_bounded_scalable_equal(
    %arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>,
    %arg2: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index

  %0 = scf.for %iv = %c0 to %c256 step %c8_vscale iter_args(%out = %arg2) -> (tensor<256x128xf32>) {
    %sz = affine.min affine_map<(d0)[s0] -> (-d0 + 256, s0)>(%iv)[%c8_vscale]
    %ext_out = tensor.extract_slice %out[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_a = tensor.extract_slice %arg0[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %ext_b = tensor.extract_slice %arg1[%iv, 0] [%sz, 128] [1, 1]
        : tensor<256x128xf32> to tensor<?x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<?x128xf32>, tensor<?x128xf32>)
        outs(%ext_out : tensor<?x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<?x128xf32>
    %inserted = tensor.insert_slice %computed into %out[%iv, 0] [%sz, 128] [1, 1]
        : tensor<?x128xf32> into tensor<256x128xf32>
    scf.yield %inserted : tensor<256x128xf32>
  }

  %pack = linalg.pack %0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // d0 tiled by 8*vscale == inner tile -> Equal; d1 untiled -> Unknown.
    %a, %b = transform.test.fuse_consumer %pack into (%loop) {inner_tile_alignments = array<i64: 2, 0>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Pack as a PRODUCER / standalone tiling ignores the alignment hint (only the
// consumer-fusion path consults it). The loop tile (16*vscale) is 2x the inner
// tile (8*vscale) and the hint lies that they are `Equal`; the producer path
// must ignore it, so the outer dims stay dynamic (ceilDiv) and do NOT collapse
// to 1. This pins the no-op quadrant of the hint matrix.

// CHECK-LABEL: func.func @pack_producer_ignores_hint
//      CHECK:    %[[PACK:.*]] = linalg.pack
// CHECK-SAME:        tensor<?x?xf32> -> tensor<?x?x?x?xf32>
func.func @pack_producer_ignores_hint(
    %src: tensor<256x128xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %0 = linalg.pack %src inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, %c4_vscale]
      into %dest : tensor<256x128xf32> -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // tile = 2x the inner tile; the `Equal` hint is a deliberate lie -- the
    // producer/standalone path ignores it (outer dims stay dynamic, not 1).
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [[16], [8]]
        {inner_tile_alignments = array<i64: 2, 2>}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
