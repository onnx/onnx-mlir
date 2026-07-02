/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- OpFusionHelper.hpp - ZHigh Fusion Helper Functions ----------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// ZHigh-specific fusion subclass built on top of the generic FusionOpChain
// base class (src/Dialect/ONNX/ONNXOps/FusionOpChain.hpp).
//
// Convention: all zhigh related fusion should use a "zhigh." prefixed kind
// name, to facilitate the lowering of fused ops.
//
// -- Fusion pass (pattern creation) ------------------------------------------
//
//   ExtLayoutTransformFusion fusion;
//   if (!fusion.detectIfBeneficial(dimAnalysis, layoutTransformOp))
//     return failure();
//
//   fusion.fuse(rewriter, loc);
//
// -- Lowering pass (code generation) ------------------------------------------
//
//   ExtLayoutTransformFusion fusion;
//   fusion.retrieveOpsAndOutputValues(fusedOp);
//
//   if (!fusion.verifyAndRetrieveAttrs(fusedOp))
//     return rewriter.notifyMatchFailure(fusedOp, "pattern altered");
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZHIGH_OP_FUSION_HELPER_H
#define ONNX_MLIR_ZHIGH_OP_FUSION_HELPER_H

#include <optional>
#include <string>

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps/FusionOpChain.hpp"

namespace onnx_mlir {
namespace zhigh {

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

class ExtLayoutTransformFusion : public onnx_mlir::FusionOpChain {
public:
  static constexpr llvm::StringLiteral kKind{"zhigh.extended_layout_transform"};

  // -- Kind-specific parameters (raw C++ values) -----------------------------
  int64_t reshapeSplitAxis = -1;  ///< axis split by step-2 Reshape (-1=absent)
  int64_t reshapeSplitFactor = 1; ///< static size of second split fragment
  int64_t reshapeMergeAxis = -1;  ///< axis merged by step-4 Reshape (-1=absent)
  std::optional<mlir::ArrayAttr> transposePattern; ///< perm of step-3 Transpose
  bool dlf16ToF32 = false; ///< true when step-5 is DLF16=>F32
  std::optional<mlir::StringAttr> finalLayout; ///< target layout for step-5a LT

  // -- Non-virtual public methods ---------------------------------------------

  /// Detect and parameterize the extended layout transform chain.
  /// Resets ops, finalResults, and all param fields on entry.
  /// Calls FusionOpChain::isInsideFusedOp() first to guard against infinite
  /// rewrite loops (ops are moved, not erased, so patterns can re-match).
  /// \p dimAnalysis must be non-null.
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
