/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ FusionOpHelper.hpp - ONNXFusedOp builder base ----------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// FusionOpKindHelper is the generic base class for building and consuming
// ONNXFusedOp regions.  It owns the op list and output values that every
// fusion pattern populates, plus the three non-virtual template methods that
// encode the canonical calling sequences for the fusion pass and the lowering
// pass.
//
// -- Guard against infinite loops --------------------------------------------
//
// Fusion passes rewrite matched op chains into an ONNXFusedOp but do NOT
// erase the original ops — they are moved into the FusedOp body instead.
// Because the same pattern can therefore match the original ops a second time
// (now living inside the body), every subclass detectIfBeneficial() override
// MUST call the base-class helper as its very first check:
//
//   if (isInsideFusedOp(startOp))
//     return false; // already inside a fused op body — skip to avoid loop
//
// Without this guard the rewrite pattern would fire repeatedly on the same
// ops and the pass would diverge.
//
// -- Fusion pass (pattern creation) ------------------------------------------
//
//   MyFusion fusion;
//   if (!fusion.detect(...))
//     return failure();
//   if (!fusion.computeInputsAndInsertionPoint())
//     return failure(); // no valid single insertion point exists -- decline
//
//   fusion.fuse(rewriter, loc);
//   // => uses the insertion point cached by computeInputsAndInsertionPoint().
//   // => private create(): builds body, embedAttrs() stores params as attrs.
//   // => private replaceAndErase(): back-to-front.
//
// computeInputsAndInsertionPoint() is called automatically by
// FusedPatternForOpKind::matchAndRewrite (FusionOpBasePattern.hpp) right
// after detectIfBeneficial() succeeds -- subclasses do not need to call it
// themselves. It exists as a separate step (rather than being folded into
// fuse()) because it can fail: it is a pure query (no IR mutation) that
// determines whether a single insertion point exists which is simultaneously
// after every external input this->ops needs, and before every use -- outside
// this->ops -- of every value in finalResults. Without this check, an
// intermediate chain value with an outside use that itself (transitively)
// feeds one of this->ops' own external inputs would make ops.back() an
// unsafe insertion point -- the FusedOp's own input would end up depending on
// its own not-yet-defined output. See FusionOpHelper.cpp for the algorithm.
//
// -- Lowering pass (code generation) -----------------------------------------
//
//   MyFusion fusion;
//   fusion.retrieveOpsAndOutputValues(fusedOp);
//   // => Walks the body block.
//
//   if (!fusion.verifyAndRetrieveAttrs(fusedOp))
//     return rewriter.notifyMatchFailure(fusedOp, "pattern altered");
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_FUSION_OP_HELPER_H
#define ONNX_MLIR_FUSION_OP_HELPER_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Location;
class Operation;
class PatternRewriter;
} // namespace mlir

#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// FusionOpKindHelper  - generic base, never instantiated directly.
//===----------------------------------------------------------------------===//

class FusionOpKindHelper {
public:
  /// Chain ops in chain order: ops[i]'s output feeds ops[i+1] as an input,
  /// and ops.back() is the last op whose result becomes the FusedOp output.
  llvm::SmallVector<mlir::Operation *> ops;

  /// Values yielded by the body, one per ONNXFusedOp result.
  llvm::SmallVector<mlir::Value> finalResults;

  virtual ~FusionOpKindHelper() = default;

  // -- Non-virtual template methods (calling sequences) ----------------------

  /// Build the FusedOp, replace the original chain ops with its outputs, and
  /// erase the chain ops.  computeInputsAndInsertionPoint() must have already
  /// been called successfully (it sets the rewriter insertion point
  /// internally, at the cached anchor -- normally FusedPatternForOpKind calls
  /// it automatically; see the file header comment).
  mlir::ONNXFusedOp fuse(mlir::PatternRewriter &rewriter, mlir::Location loc);

  /// Pure query (no IR mutation): computes the FusedOp's external input list
  /// and determines whether a single insertion point exists that is
  /// simultaneously after every one of those inputs' definitions and before
  /// every outside (non-chain) use of every value in finalResults. Caches the
  /// input list and insertion point on success for fuse()/createFusedOp() to
  /// reuse. Returns false -- meaning the match must be treated as a failure,
  /// do not call fuse() -- when no such point exists. Must be called exactly
  /// once, after this->ops and this->finalResults are fully populated (i.e.
  /// after detectIfBeneficial() would otherwise return true) and before
  /// fuse().
  bool computeInputsAndInsertionPoint();

  /// Walk fusedOp.getBody().front(): collect non-YieldOp ops => this->ops and
  /// YieldOp operands => this->finalResults.  Resets both fields on entry.
  void retrieveOpsAndOutputValues(mlir::ONNXFusedOp fusedOp);

  /// Template method: calls the virtual retrieveAttrs() then verify().
  /// Returns false (and emits LLVM_DEBUG) on any failure.
  /// this->ops must already be populated (call retrieveOpsAndOutputValues
  /// first).
  bool verifyAndRetrieveAttrs(mlir::ONNXFusedOp fusedOp);

  /// Fallback: inline the FusedOp body back into the enclosing function.
  /// Call this when a dedicated lowering cannot proceed (verify() failed,
  /// or the kind has no lowering yet).  Uses the original pre-conversion
  /// FusedOp inputs so that block-argument types are preserved; the caller's
  /// PatternRewriter then converts the newly exposed ops in the same pass.
  /// Always returns LogicalResult::success().
  /// Static so that callers without a FusionOpKindHelper instance (e.g. the
  /// generic FusedOpInlineFallback catch-all) can invoke it directly.
  /// Per-kind lowerings that do have an instance can still call it as
  /// fusion.unFuse(...) — calling a static method via an instance
  /// is valid C++.
  static mlir::LogicalResult unFuse(
      mlir::PatternRewriter &rewriter, mlir::ONNXFusedOp fusedOp);

protected:
  // -- Helper for subclass detect methods ------------------------------------

  /// Returns true when \p op is directly nested inside an ONNXFusedOp body.
  /// Call this first in every detectIfBeneficial() override to avoid infinite
  /// rewrite loops (see the "Guard against infinite loops" note above).
  static bool isInsideFusedOp(mlir::Operation *op);

  // -- Pure-virtual subclass contract ----------------------------------------

  /// Returns the kind string that identifies this pattern on the ONNXFusedOp.
  virtual llvm::StringRef getKind() const = 0;

  /// Write all subclass param fields to named MLIR attributes on fusedOp.
  virtual void embedAttrs(mlir::ONNXFusedOp fusedOp) const = 0;

  /// Read all named MLIR attributes from fusedOp back into the subclass param
  /// fields.  Returns false if any required attribute is absent.
  virtual bool retrieveAttrs(mlir::ONNXFusedOp fusedOp) = 0;

  /// Cross-check this->ops against the param fields read by retrieveAttrs().
  /// Returns false when the body no longer matches the stored parameters.
  virtual bool verify() const = 0;

  // -- Additional subclass contract member (not a virtual) -------------------
  //
  // Every subclass must also define:
  //
  //   bool detectIfBeneficial(const DimAnalysis *dimAnalysis, AnchorOpType
  //   startOp);
  //
  // where AnchorOpType is the op the subclass anchors its match on (e.g.
  // ONNXLayoutTransformOp, ONNXUnsqueezeOp).  It cannot be declared here as a
  // virtual: AnchorOpType differs per subclass, and virtual dispatch requires
  // a uniform signature across overrides.  Instead it is enforced at compile
  // time wherever the subclass is plugged into a pattern, e.g.
  // FusedPatternForOpKind<AnchorOpType, FusionT> (see FusionOpBasePattern.hpp)
  // — omitting detectIfBeneficial fails to compile at that instantiation, not
  // here.  Must call isInsideFusedOp(startOp) first (see above).

private:
  // -- Cache populated by computeInputsAndInsertionPoint(), consumed by
  // fuse()/createFusedOp(). Not meaningful until that call has returned true.
  llvm::SmallVector<mlir::Value> cachedFusedInputs;
  mlir::Operation *cachedInsertionAnchor = nullptr;
  // true: insert immediately after cachedInsertionAnchor (it is the latest
  // external input's defining op). false: insert immediately before
  // cachedInsertionAnchor (the old ops.back()-relative default, still valid
  // whenever no outside use of a yielded value forces an earlier point).
  bool cachedInsertAfterAnchor = false;

  /// Build the ONNXFusedOp body — called by fuse().
  mlir::ONNXFusedOp create(mlir::PatternRewriter &rewriter, mlir::Location loc);

  /// Replace output ops and erase internal ops — called by fuse().
  void replaceAndErase(
      mlir::PatternRewriter &rewriter, mlir::ONNXFusedOp fusedOp);

  /// Body-building implementation used by create().
  mlir::ONNXFusedOp createFusedOp(mlir::PatternRewriter &rewriter,
      mlir::Location loc, llvm::StringRef kind);
};

} // namespace onnx_mlir

#endif // ONNX_MLIR_FUSION_OP_HELPER_H
