// RUN: mlir-opt %s -transform-interpreter -canonicalize -cse -split-input-file --verify-diagnostics | FileCheck %s

// A 3-op dispatch tiled from the mmt4d root, fusing its consumers, with an
// inner-tile alignment hint driving the scalable unpack fusion:
//
//   linalg.mmt4d              (root, produces a packed [M, N, M0, N0] layout)
//   -> linalg.generic         (transposing bias add: [M, N, M0, N0] -> [N, M, N0, M0])
//   -> linalg.unpack          ([N, M, N0, M0] -> [N*N0, M*M0])
//
// The mmt4d is tiled along its scalable inner dim N0 (iteration dim 4) by
// 8*vscale; the generic and unpack are fused as consumers. The loop tile
// (8*vscale) and the unpack inner tile (8*vscale) are both scalable, so their
// relationship is not statically decidable and the unpack fusion needs the
// `inner_tile_alignments` hint (supplied to the SCF driver as a control
// function). The transposing bias add moves the scalable N0 inner tile onto the
// unpack's dest dim 0, so `Equal` must sit at index 0 (`array<i64: 2, 0>`); the
// generic ignores the hint. The negative case below shows the hint is
// load-bearing.

#id4 = affine_map<(m, n, m0, n0) -> (m, n, m0, n0)>
#tr4 = affine_map<(m, n, m0, n0) -> (n, m, n0, m0)>

// CHECK: #[[$TR:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-LABEL: func.func @mmt4d_transpose_unpack
// CHECK-SAME:      %[[LHS:.+]]: tensor<2x2x4x2xf32>, %[[RHS:.+]]: tensor<2x2x?x2xf32>, %[[ACC:.+]]: tensor<2x2x4x?xf32>, %[[BIAS:.+]]: tensor<2x2x?x4xf32>, %[[TRINIT:.+]]: tensor<2x2x?x4xf32>, %[[OUT:.+]]: tensor<?x8xf32>
//      CHECK:    %[[VSCALE:.*]] = vector.vscale
//      CHECK:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %{{.*}} : index
//      CHECK:    %[[RES:.*]]:3 = scf.for {{.*}} step %[[C8_VSCALE]]
// CHECK-SAME:        iter_args(%{{.*}} = %[[ACC]], %{{.*}} = %[[TRINIT]], %{{.*}} = %[[OUT]])
//      CHECK:      %[[MM:.*]] = linalg.mmt4d
//      CHECK:      %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:          indexing_maps = [#{{.+}}, #[[$TR]], #[[$TR]]]
// CHECK-SAME:          ins(%[[MM]],
//      CHECK:      %[[UNPACK:.*]] = linalg.unpack %[[GENERIC]]
// CHECK-SAME:          outer_dims_perm = [0, 1] inner_dims_pos = [0, 1]
// CHECK-SAME:          inner_tiles = [%[[C8_VSCALE]], 4]
//      CHECK:      scf.yield {{.*}}, {{.*}}, %{{.*}} :
//      CHECK:    return %[[RES]]#2
func.func @mmt4d_transpose_unpack(
    %lhs: tensor<2x2x4x2xf32>, %rhs: tensor<2x2x?x2xf32>,
    %acc: tensor<2x2x4x?xf32>, %bias: tensor<2x2x?x4xf32>,
    %trinit: tensor<2x2x?x4xf32>, %out: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index

  // 1. mmt4d root: lhs[M,K,M0,K0] x rhs[N,K,N0,K0] -> out[M,N,M0,N0].
  %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x2xf32>, tensor<2x2x?x2xf32>)
      outs(%acc : tensor<2x2x4x?xf32>) -> tensor<2x2x4x?xf32>

  // 2. transposing bias add: [M,N,M0,N0] -> [N,M,N0,M0].
  %tr = linalg.generic {
      indexing_maps = [#id4, #tr4, #tr4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%mm, %bias : tensor<2x2x4x?xf32>, tensor<2x2x?x4xf32>)
      outs(%trinit : tensor<2x2x?x4xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<2x2x?x4xf32>

  // 3. unpack: [N,M,N0,M0] -> [N*N0, M*M0] = [?, 8].
  %unpack = linalg.unpack %tr outer_dims_perm = [0, 1] inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, 4] into %out
      : tensor<2x2x?x4xf32> -> tensor<?x8xf32>
  return %unpack : tensor<?x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mmt4d = transform.structured.match ops{["linalg.mmt4d"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // Tile the mmt4d root along its scalable inner dim N0 (iteration dim 4).
    %tiled, %loop = transform.structured.tile_using_for %mmt4d tile_sizes [0, 0, 0, 0, [8], 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Fuse the transposing bias add (ignores the hint).
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fg, %loop2 = transform.test.fuse_consumer %gen into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Fuse the unpack. The transpose puts the tiled scalable N0 inner tile on
    // dest dim 0, so Equal sits at index 0.
    %unp = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fu, %loop3 = transform.test.fuse_consumer %unp into (%loop2) {inner_tile_alignments = array<i64: 2, 0>}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Negative (hint is load-bearing): same chain fused without any alignment hint.
// The loop tile (8*vscale) and the unpack inner tile (8*vscale) are both
// scalable, so the relationship is not statically decidable and fusion of the
// unpack fails.

#id4 = affine_map<(m, n, m0, n0) -> (m, n, m0, n0)>
#tr4 = affine_map<(m, n, m0, n0) -> (n, m, n0, m0)>

func.func @mmt4d_transpose_unpack_no_hint(
    %lhs: tensor<2x2x4x2xf32>, %rhs: tensor<2x2x?x2xf32>,
    %acc: tensor<2x2x4x?xf32>, %bias: tensor<2x2x?x4xf32>,
    %trinit: tensor<2x2x?x4xf32>, %out: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index

  %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x2xf32>, tensor<2x2x?x2xf32>)
      outs(%acc : tensor<2x2x4x?xf32>) -> tensor<2x2x4x?xf32>

  %tr = linalg.generic {
      indexing_maps = [#id4, #tr4, #tr4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%mm, %bias : tensor<2x2x4x?xf32>, tensor<2x2x?x4xf32>)
      outs(%trinit : tensor<2x2x?x4xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<2x2x?x4xf32>

  // expected-error @below {{'linalg.unpack' op failed to fuse consumer of slice}}
  %unpack = linalg.unpack %tr outer_dims_perm = [0, 1] inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, 4] into %out
      : tensor<2x2x?x4xf32> -> tensor<?x8xf32>
  return %unpack : tensor<?x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mmt4d = transform.structured.match ops{["linalg.mmt4d"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %mmt4d tile_sizes [0, 0, 0, 0, [8], 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fg, %loop2 = transform.test.fuse_consumer %gen into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %unp = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fu, %loop3 = transform.test.fuse_consumer %unp into (%loop2)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
