// RUN: mlir-opt %s -transform-interpreter -canonicalize -cse -split-input-file --verify-diagnostics | FileCheck %s

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
