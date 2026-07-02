/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ FusionOpChain.hpp - ONNXFusedOp builder base -----------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// FusionOpChain is the generic base class for building and consuming
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
//
//   fusion.fuse(rewriter, loc);
//   // => sets insertion point to ops.back() internally.
//   // => private create(): builds body, embedAttrs() stores params as attrs.
//   // => private replaceAndErase(): back-to-front.
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

#ifndef ONNX_MLIR_FUSION_OP_CHAIN_H
#define ONNX_MLIR_FUSION_OP_CHAIN_H

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
// FusionOpChain  - generic base, never instantiated directly.
//===----------------------------------------------------------------------===//

class FusionOpChain {
public:
  /// Chain ops in chain order: ops[i]'s output feeds ops[i+1] as an input,
  /// and ops.back() is the last op whose result becomes the FusedOp output.
  llvm::SmallVector<mlir::Operation *> ops;

  /// Values yielded by the body, one per ONNXFusedOp result.
  llvm::SmallVector<mlir::Value> finalResults;

  virtual ~FusionOpChain() = default;

  // -- Non-virtual template methods (calling sequences) ----------------------

  /// Build the FusedOp, replace the original chain ops with its outputs, and
  /// erase the chain ops.  The caller must set the rewriter insertion point
  /// before calling this (typically just before ops.back()).
  mlir::ONNXFusedOp fuse(mlir::PatternRewriter &rewriter, mlir::Location loc);

  /// Walk fusedOp.getBody().front(): collect non-YieldOp ops => this->ops and
  /// YieldOp operands => this->finalResults.  Resets both fields on entry.
  void retrieveOpsAndOutputValues(mlir::ONNXFusedOp fusedOp);

  /// Template method: calls the virtual retrieveAttrs() then verify().
  /// Returns false (and emits LLVM_DEBUG) on any failure.
  /// this->ops must already be populated (call retrieveOpsAndOutputValues
  /// first).
  bool verifyAndRetrieveAttrs(mlir::ONNXFusedOp fusedOp);

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

private:
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

#endif // ONNX_MLIR_FUSION_OP_CHAIN_H
