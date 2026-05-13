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
    %a, %b = transform.test.fuse_consumer %pack into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion - static producer step 64 is a power-of-2 multiple of the
// pack inner tile size (8*vscale), but not equal. Fusion succeeds via the
// aligned (non-equal) path: the equality flag does not fire and the outer
// dim of the fused pack tile is dynamic (`64 ceildiv 8*vscale`).

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
    %a, %b = transform.test.fuse_consumer %pack into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion - both producer step and pack inner tile are scalable,
// step (8*vscale) is a power-of-2 multiple of the inner tile (4*vscale) but
// not equal (vscale multipliers 8 vs 4). The equality flag must NOT fire;
// the outer dim of the fused pack tile stays dynamic
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
    %a, %b = transform.test.fuse_consumer %pack into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion (negative): linalg.pack with scalable inner tiles where the
// loop step (12*vscale) is not aligned to the pack inner tile size (8*vscale).

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
