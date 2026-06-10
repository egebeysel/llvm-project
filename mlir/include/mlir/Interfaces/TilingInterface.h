//===- TilingInterface.h - Interface for tiling operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the TilingInterface defined in
// `TilingInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_TILINGINTERFACE_H_
#define MLIR_INTERFACES_TILINGINTERFACE_H_

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

/// Container for result values of tiling.
/// - `tiledOps` contains operations created by the tiling implementation that
///   are returned to the caller for further transformations.
/// - `tiledValues` contains the tiled value corresponding to the result of the
///   untiled operation.
/// - `generatedSlices` contains the list of slices that are generated during
///   tiling. These slices can be used for fusing producers.
struct TilingResult {
  SmallVector<Operation *> tiledOps;
  SmallVector<Value> tiledValues;
  SmallVector<Operation *> generatedSlices;
};

/// Tiling can be thought of as splitting a dimension into 2 and
/// materializing the outer dimension as a loop:
///
/// op[original] -> op[original / x, x] -> loop[original] { op[x] }
///
/// For parallel dimensions, the split can only happen in one way, with both
/// dimensions being parallel. For reduction dimensions however, there is a
/// choice in how we split the reduction dimension. This enum exposes this
/// choice.
enum class ReductionTilingStrategy {
  // [reduction] -> [reduction1, reduction2]
  // -> loop[reduction1] { [reduction2] }
  FullReduction,
  // [reduction] -> [reduction1, parallel2]
  // -> loop[reduction1] { [parallel2] }; merge[reduction1]
  PartialReductionOuterReduction,
  // [reduction] -> [parallel1, reduction2]
  // -> loop[parallel1] { [reduction2] }; merge[parallel1]
  PartialReductionOuterParallel
};

/// Container for the result of merge operation of tiling.
/// - `mergeOps` contains operations created during the merge.
/// - `replacements` contains the values that represents the result of the
/// merge. These are used as replacements for the original tiled operation.
struct MergeResult {
  SmallVector<Operation *> mergeOps;
  SmallVector<Value> replacements;
};

/// Per-dimension alignment of a loop tile size to a `linalg.pack` /
/// `linalg.unpack` inner tile size, supplied by the caller (which performed the
/// tiling and knows both the tile sizes and the inner tiles) so that
/// pack/unpack TilingInterface implementations need not re-derive it from the
/// materialized IR. An absent entry (or `Unknown`) means "no information": the
/// implementation must fall back to its prior behavior for that dimension.
///   - `Multiple`: the loop tile size is an integer multiple of the inner tile.
///   - `Equal`:    the loop tile size equals the inner tile size.
///
/// This is a caller assertion, not a checked fact: it is only consulted when
/// the relationship cannot be decided from the IR (e.g., scalable or dynamic
/// sizes). When the tile and inner-tile sizes are both statically known,
/// implementations trust that static comparison instead, so a hint that
/// contradicts statically known sizes is ignored rather than allowed to produce
/// incorrect tiling. The hint is otherwise never verified, so an incorrect
/// assertion produces silently invalid tiling.
///
/// Entries are indexed by the dimensions the consulting method reasons about,
/// i.e. the op's iteration domain (in pre-interchange order -- `interchange`
/// reorders the generated loops only). This also holds for the
/// `*FromOperandTiles` consumer-fusion methods: for a pack the iteration domain
/// coincides with the unpacked operand's source dimensions, while for an unpack
/// the entry for the i-th inner tile sits at its dest dimension
/// `inner_dims_pos[i]` (a dimension of the unpacked tensor, not of the packed
/// operand). Entries are not remapped through indexing maps or
/// `outer_dims_perm` (for a transposing pack they stay in source order,
/// pre-permutation), so the caller must pre-arrange them to match that order.
enum class InnerTileAlignment : int64_t { Unknown = 0, Multiple, Equal };

/// Returns true iff `value` is a valid `InnerTileAlignment` enumerator.
inline bool isValidInnerTileAlignment(int64_t value) {
  switch (static_cast<InnerTileAlignment>(value)) {
  case InnerTileAlignment::Unknown:
  case InnerTileAlignment::Multiple:
  case InnerTileAlignment::Equal:
    return true;
  }
  return false;
}

/// Verifies that every entry of a raw `inner_tile_alignments` integer array is
/// a valid `InnerTileAlignment`, emitting the standard op error on `op`
/// otherwise.
LogicalResult verifyInnerTileAlignments(Operation *op,
                                        ArrayRef<int64_t> alignments);

/// Maps a validated `inner_tile_alignments` integer array onto the
/// per-dimension `InnerTileAlignment` hints consumed by the tiling driver.
SmallVector<InnerTileAlignment>
convertInnerTileAlignments(ArrayRef<int64_t> alignments);

} // namespace mlir

/// Include the ODS generated interface header files.
#include "mlir/Interfaces/TilingInterface.h.inc"

#endif // MLIR_INTERFACES_TILINGINTERFACE_H_
