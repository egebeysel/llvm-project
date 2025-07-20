// RUN: mlir-opt %s -transform-interpreter -canonicalize -cse -split-input-file | FileCheck %s

// Test 1: Perfect scalable tiling - scalable tile sizes equal scalable inner
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
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// Test 2: Aligned scalable tiling - static tile sizes aligned to scalable
// inner tiles. Tile size 64 with inner tile 8*vscale: 64/8 = 8 is a power of
// 2, so aligned. Tile size 32 with inner tile 4*vscale: 32/4 = 8 likewise.

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
      %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [64, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// ============================================================================
// Test 3: Producer fusion - unpack with scalable inner tiles fused as a
// producer into the consumer (linalg.exp) tiled with aligned static tile
// sizes [64, 32].
// ============================================================================

// CHECK-LABEL: func.func @unpack_elemwise_scalable
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             linalg.exp ins(%[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @unpack_elemwise_scalable(%arg0: tensor<4x8x?x?xf32>, %arg1: tensor<?x?xf32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
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
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [64, 32] interchange [0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// ============================================================================
// Test 4: Producer fusion - unpack with scalable inner tiles fused as a
// producer into the consumer (linalg.exp) tiled with perfect tiling.
// ============================================================================

// CHECK-LABEL: func.func @unpack_elemwise_scalable
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:            tensor<1x1x?x?xf32> -> tensor<?x?xf32>
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             linalg.exp ins(%[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @unpack_elemwise_scalable(%arg0: tensor<4x8x?x?xf32>, %arg1: tensor<?x?xf32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
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
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [[8], [4]] interchange [0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}
