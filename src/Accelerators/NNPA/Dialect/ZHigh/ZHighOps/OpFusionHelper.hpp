/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- OpFusionHelper.hpp - ZHigh Fusion Helper Functions ----------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Shared helpers for recognizing and parametrization fused ZHigh operation
// patterns.  The FusionOpChain base class provides generic body-building and
// retrieval mechanics; one subclass per ONNXFusedOp kind carries the
// pattern-specific parameters and implements the four virtual methods.
//
// -- Fusion pass (pattern creation) ------------------------------------------
//
//   ExtLayoutTransformFusion fusion;
//   if (!fusion.detectIfBeneficial(dimAnalysis, layoutTransformOp))
//     return failure();
//
//   // Set insertion point before the last chain op so all inputs dominate it.
//   rewriter.setInsertionPoint(fusion.finalResults[0].getDefiningOp());
//   ONNXFusedOp fusedOp = fusion.create(rewriter, loc);
//   // => FusionOpChain::create: scans fusion.ops for inputs, builds isolated
//   //    body, clones ops, emits YieldOp, then calls embedAttrs() to store
//   //    all parameters as named MLIR attributes on the FusedOp.
//
//   // Erase the original chain ops back-to-front.
//   rewriter.replaceOp(fusion.ops.back(), fusedOp.getOutputs()[0]);
//   for (int i = (int)fusion.ops.size() - 2; i >= 0; --i)
//     rewriter.eraseOp(fusion.ops[i]);
//
// -- Lowering pass (code generation) ------------------------------------------
//
//   ExtLayoutTransformFusion fusion;
//   fusion.retrieveOpsAndOutputValues(fusedOp);
//   // => Walks the body block: non-Yield ops => fusion.ops,
//   //    YieldOp operands => fusion.finalResults.
//
//   if (!fusion.verifyAndRetrieveAttrs(fusedOp))
//     return rewriter.notifyMatchFailure(fusedOp, "pattern altered");
//   // => Calls retrieveAttrs() to read stored params into fusion fields,
//   //    then verify() to cross-check fusion.ops against those params.
//   //    Returns false if an optimisation pass altered the body; the generic
//   //    FusedOpInliningPattern then handles the fallback.
//
//   // Use fusion.reshapeSplitAxis, .reshapeSplitFactor, etc. directly for
//   // Krnl code generation.  Use fusion.ops[i] for richer type information
//   // when needed (e.g. intermediate tensor shapes not visible on the FusedOp
//   // result type after shape propagation).
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZHIGH_OP_FUSION_HELPER_H
#define ONNX_MLIR_ZHIGH_OP_FUSION_HELPER_H

#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class PatternRewriter;
} // namespace mlir

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// FusionOpChain  - generic base, never instantiated directly.
//
// Owns the op list and output values that every pattern populates, plus the
// three non-virtual template methods that encode the canonical calling
// sequences for the fusion pass and the lowering pass.
//===----------------------------------------------------------------------===//

class FusionOpChain {
public:
  /// Chain ops in order, populated by detectIfBeneficial() or
  /// retrieveOpsAndOutputValues().
  llvm::SmallVector<mlir::Operation *> ops;

  /// Values yielded by the body, one per ONNXFusedOp result.  Populated by
  /// detectIfBeneficial() (live IR values) or retrieveOpsAndOutputValues()
  /// (YieldOp operands inside the body).
  llvm::SmallVector<mlir::Value> finalResults;

  virtual ~FusionOpChain() = default;

  // -- Non-virtual template methods (calling sequences) ----------------------

  /// Build the ONNXFusedOp: scans this->ops to classify external operands
  /// (constants => cloned inside the body; others => block arguments), builds
  /// the isolated body region, clones the chain ops, emits YieldOp over
  /// this->finalResults, calls the virtual embedAttrs(), and returns the op.
  /// The caller is responsible for setting the rewriter insertion point before
  /// calling this method.
  mlir::ONNXFusedOp create(
      mlir::PatternRewriter &rewriter, mlir::Location loc);

  /// Walk fusedOp.getBody().front(): collect non-YieldOp ops => this->ops and
  /// YieldOp operands => this->finalResults.  Resets both fields on entry.
  /// Cannot fail for a well-formed FusedOp body.
  void retrieveOpsAndOutputValues(mlir::ONNXFusedOp fusedOp);

  /// Template method: calls the virtual retrieveAttrs() then verify().
  /// Returns false (and emits LLVM_DEBUG) on any failure.
  /// this->ops must already be populated (call retrieveOpsAndOutputValues first).
  bool verifyAndRetrieveAttrs(mlir::ONNXFusedOp fusedOp);

protected:
  // -- Pure-virtual subclass contract ----------------------------------------

  /// Returns the kind string that identifies this pattern on the ONNXFusedOp.
  virtual llvm::StringRef getKind() const = 0;

  /// Write all subclass param fields to named MLIR attributes on fusedOp.
  /// This is the ONLY function that writes attrs.
  virtual void embedAttrs(mlir::ONNXFusedOp fusedOp) const = 0;

  /// Read all named MLIR attributes from fusedOp back into the subclass param
  /// fields.  Returns false if any required attribute is absent.
  /// This is the ONLY function that reads attrs.
  virtual bool retrieveAttrs(mlir::ONNXFusedOp fusedOp) = 0;

  /// Cross-check this->ops (already populated) against the param fields just
  /// read by retrieveAttrs().  Returns false when the body no longer matches
  /// the stored parameters (e.g. an optimisation pass altered a body op).
  virtual bool verify() const = 0;

private:
  /// Internal body-building helper called by create().
  mlir::ONNXFusedOp createFusedOp(mlir::PatternRewriter &rewriter,
      mlir::Location loc, llvm::StringRef kind);
};

//===----------------------------------------------------------------------===//
// ExtLayoutTransformFusion
//
// Subclass for ONNXFusedOp(kind = "zhigh.extended_layout_transform").
//
// Pattern:
//   ONNXLayoutTransformOp           ZTensor => CPU         (required)
//   ONNXReshapeOp    (optional)     split one dim into two
//   ONNXTransposeOp  (optional)     permute; last dim must stay in place
//   ONNXReshapeOp    (optional)     merge two dims into one
//   ONNXLayoutTransformOp  (opt.)   CPU => ZTensor          (step 5a)
//     OR ZHighDLF16ToF32Op (opt.)   DLF16 => F32            (step 5b)
//===----------------------------------------------------------------------===//

class ExtLayoutTransformFusion : public FusionOpChain {
public:
  static constexpr llvm::StringLiteral kKind{
      "zhigh.extended_layout_transform"};

  // -- Kind-specific parameters (raw C++ values) -----------------------------
  int64_t reshapeSplitAxis = -1;   ///< axis split by step-2 Reshape (-1=absent)
  int64_t reshapeSplitFactor = 1;  ///< static size of second split fragment
  int64_t reshapeMergeAxis = -1;   ///< axis merged by step-4 Reshape (-1=absent)
  std::optional<mlir::ArrayAttr> transposePattern; ///< perm of step-3 Transpose
  bool dlf16ToF32 = false;         ///< true when step-5 is DLF16=>F32
  std::optional<mlir::StringAttr> finalLayout;     ///< target layout for step-5a LT

  // -- Non-virtual public methods ---------------------------------------------

  /// Detect and parameterize the extended layout transform chain.
  /// Resets ops, finalResults, and all param fields on entry.
  /// \p dimAnalysis must be non-null; dim comparisons use it for full precision.
  /// Returns true (and populates all fields) only when the chain passes all
  /// validation and the beneficial threshold.
  bool detectIfBeneficial(
      const DimAnalysis *dimAnalysis, mlir::ONNXLayoutTransformOp startOp);

  // -- Virtual overrides ------------------------------------------------------
  llvm::StringRef getKind() const override { return kKind; }
  void embedAttrs(mlir::ONNXFusedOp fusedOp) const override;
  bool retrieveAttrs(mlir::ONNXFusedOp fusedOp) override;
  bool verify() const override;
};

} // namespace zhigh
} // namespace onnx_mlir

#endif // ONNX_MLIR_ZHIGH_OP_FUSION_HELPER_H
