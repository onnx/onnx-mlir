/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ONNXFusionOpHelper.hpp - ONNXFusedOp CPU kinds ---------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Shared home for CPU-side (non-accelerator) FusionOpKindHelper subclasses --
// the counterpart of ZHighFusionOpHelper.{hpp,cpp}, which plays this role for
// NNPA-specific kinds. More general fusion kinds are expected to land here
// over time; this file is not named after any one pattern.
//
// -- Fusion pass (pattern creation) ------------------------------------------
//
//   SplitOpGatherFusionHelper fusion;
//   if (!fusion.detectIfBeneficial(dimAnalysis, concatOp))
//     return failure();
//
//   fusion.fuse(rewriter, loc);
//
// -- Lowering pass (code generation) ------------------------------------------
//
//   SplitOpGatherFusionHelper fusion;
//   fusion.retrieveOpsAndOutputValues(fusedOp);
//
//   if (!fusion.verifyAndRetrieveAttrs(fusedOp))
//     return rewriter.notifyMatchFailure(fusedOp, "pattern altered");
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_FUSION_OP_HELPER_H
#define ONNX_MLIR_ONNX_FUSION_OP_HELPER_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/Transforms/FusionOpHelper.hpp"

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// SplitOpGatherFusionHelper
//
// Subclass for ONNXFusedOp(kind = "simd-split-op-gather").
//
// Pattern (the RoPE "rotate_half" idiom):
//   ONNXConcatOp     exactly 2 inputs; axis is the innermost dim (required)
//   each of the 2 inputs traces back, through an optional single elementwise
//   op (unary, or binary with exactly one extra, same-shape, non-cross-half
//   operand -- absent means a plain copy of that half), to an ONNXSliceOp of
//   a shared source tensor: dense (step=1), the two slices contiguous and
//   together covering the whole of the split axis.
//
// Restrictions (v1, see wiki/plan for the full rationale):
//  - split axis must be the innermost (last) dimension;
//  - each slice's `axes` operand must name exactly the split axis (no
//    other-axis restriction to separately verify);
//  - the split point and the source tensor's split-axis dim size must both
//    be compile-time literals;
//  - a per-half op's extra operand must have exactly the same shape as that
//    half (no broadcasting), and must not be the other half's slice result
//    directly (no transitive cross-half check).
//
// Chain-op order convention (specific to this kind -- the base class
// imposes no particular order, each kind picks its own): `ops` (the
// FusionOpKindHelper base class's chain-op list, populated by
// detectIfBeneficial and re-derived by retrieveOpsAndOutputValues) is
// always laid out low-half-first --
//   [sliceLow, opLow?, sliceHigh, opHigh?, concatOp]
// -- regardless of which half happens to be Concat's first operand in the
// actual IR. This fixed order is what lets verify() re-check the chain
// positionally despite two independently optional ops (hasOpForSplitLow/
// High give 4 presence combinations, which ops.size() alone can't
// disambiguate). It also means the ops as cloned into the FusedOp body are
// printed low-then-high even when the source IR had them the other way --
// see detectIfBeneficial's "Fixed, deterministic op order" comment and
// verify()'s indexed walk, both in ONNXFusionOpHelper.cpp.
//
// Unique-use invariant: every intermediate value (each Slice result, and
// each per-half op's result) has exactly one use.
//===----------------------------------------------------------------------===//

class SplitOpGatherFusionHelper : public onnx_mlir::FusionOpKindHelper {
public:
  static constexpr llvm::StringLiteral kKind{"simd-split-op-gather"};

  // "Low"/"High" always name the two halves by their INPUT split identity
  // ([0,splitPoint) and [splitPoint,D), per `splitPoint` below) -- never by
  // output position. Every field below is suffixed ForSplitLow/ForSplitHigh
  // (or is itself named splitPoint) to keep that one coordinate system
  // explicit throughout, since Concat's own operand order is independent of
  // it (see outputOffsetForSplit{Low,High} immediately below).
  int64_t axis = -1;       ///< normalized, innermost axis of the source
  int64_t splitPoint = -1; ///< k: low = [0,k), high = [k,D)
  bool hasOpForSplitLow = false;
  bool hasOpForSplitHigh = false;

  /// Where the (possibly transformed) low/high split half is written in the
  /// output tensor, along `axis`. Exactly one of the two is always 0 and the
  /// other is the length of whichever half Concat happens to place first --
  /// which one is 0 therefore depends on Concat's operand order, not on
  /// `splitPoint`.
  int64_t outputOffsetForSplitLow = -1;
  int64_t outputOffsetForSplitHigh = -1;

  /// Detect and parameterize the split-op-gather chain, anchored on the
  /// Concat that gathers the two (possibly transformed) halves back
  /// together. Resets ops, finalResults, and all param fields on entry.
  /// \p dimAnalysis must be non-null.
  bool detectIfBeneficial(
      const DimAnalysis *dimAnalysis, mlir::ONNXConcatOp startOp);

  llvm::StringRef getKind() const override { return kKind; }
  void embedAttrs(mlir::ONNXFusedOp fusedOp) const override;
  bool retrieveAttrs(mlir::ONNXFusedOp fusedOp) override;
  bool verify() const override;
};

} // namespace onnx_mlir

#endif // ONNX_MLIR_ONNX_FUSION_OP_HELPER_H
