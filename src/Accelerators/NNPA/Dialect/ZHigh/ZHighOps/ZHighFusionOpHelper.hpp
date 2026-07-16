/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZHighFusionOpHelper.hpp - ZHigh Fusion Helper Functions -----===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// ZHigh-specific fusion subclass built on top of the generic
// FusionOpKindHelper base class
// (src/Dialect/ONNX/Transforms/FusionOpHelper.hpp).
//
// Convention: all zhigh related fusion should use a "zhigh." prefixed kind
// name, to facilitate the lowering of fused ops.
//
// -- Fusion pass (pattern creation) ------------------------------------------
//
//   ExtLayoutTransformFusionHelper fusion;
//   if (!fusion.detectIfBeneficial(dimAnalysis, layoutTransformOp))
//     return failure();
//
//   fusion.fuse(rewriter, loc);
//
// -- Lowering pass (code generation) ------------------------------------------
//
//   ExtLayoutTransformFusionHelper fusion;
//   fusion.retrieveOpsAndOutputValues(fusedOp);
//
//   if (!fusion.verifyAndRetrieveAttrs(fusedOp))
//     return rewriter.notifyMatchFailure(fusedOp, "pattern altered");
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZHIGH_FUSION_OP_HELPER_H
#define ONNX_MLIR_ZHIGH_FUSION_OP_HELPER_H

#include <optional>
#include <string>

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/Transforms/FusionOpHelper.hpp"

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// ExtLayoutTransformFusionHelper
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

class ExtLayoutTransformFusionHelper : public onnx_mlir::FusionOpKindHelper {
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
  /// Calls FusionOpKindHelper::isInsideFusedOp() first to guard against
  /// infinite rewrite loops (ops are moved, not erased, so patterns can
  /// re-match). \p dimAnalysis must be non-null. Returns true (and populates
  /// all fields) only when the chain passes all validation and the beneficial
  /// threshold.
  bool detectIfBeneficial(
      const DimAnalysis *dimAnalysis, mlir::ONNXLayoutTransformOp startOp);

  // -- Virtual overrides ------------------------------------------------------
  llvm::StringRef getKind() const override { return kKind; }
  void embedAttrs(mlir::ONNXFusedOp fusedOp) const override;
  bool retrieveAttrs(mlir::ONNXFusedOp fusedOp) override;
  bool verify() const override;
};

//===----------------------------------------------------------------------===//
// ExpandMulStickFusionHelper
//
// Subclass for ONNXFusedOp(kind = "zhigh.expand-mul-stick").
//
// Pattern:
//  ONNXUnsqueezeOp  one axis P; innermost dim of result static mod 64
//                   (required)
//  ONNXExpandOp     dim P expands from 1 to N (N static, >= 2)
//                   (required)
//  ONNXMulOp        element-wise mul by scalar F32/I32/I64 const
//                   (optional; when absent, mulScalar stays at its neutral
//                    1.f default)
//  ONNXReshapeOp    dims 0..P may collapse; dims after P unchanged (required)
//  ZHighStickOp     stick to 3D / 3DS / 4D (required)
//
// Unique-use invariant: every intermediate value (unsqueeze through reshape)
// has exactly one use.  The stick result is not checked.
//===----------------------------------------------------------------------===//

class ExpandMulStickFusionHelper : public onnx_mlir::FusionOpKindHelper {
public:
  static constexpr llvm::StringLiteral kKind{"zhigh.expand-mul-stick"};

  int64_t unsqueezedPosition = -1; ///< P: axis inserted by unsqueeze
  int64_t expansionN = -1;         ///< N: value dim P expands to
  float mulScalar = 1.f;           ///< scalar multiplier (F32; 1 = neutral)
  int64_t reshapeFirstCollapsedDim =
      -1; ///< first input dim in merge run (-1 = none)
  int64_t reshapeCollapsedCount = 0; ///< # consecutive input dims merged into 1
  std::optional<mlir::StringAttr> stickFormat; ///< "3D", "3DS", or "4D"

  /// Detect and parameterize the expand-mul-stick chain.
  /// \p dimAnalysis must be non-null.
  bool detectIfBeneficial(
      const DimAnalysis *dimAnalysis, mlir::ONNXUnsqueezeOp startOp);

  llvm::StringRef getKind() const override { return kKind; }
  void embedAttrs(mlir::ONNXFusedOp fusedOp) const override;
  bool retrieveAttrs(mlir::ONNXFusedOp fusedOp) override;
  bool verify() const override;
};

} // namespace zhigh
} // namespace onnx_mlir

#endif // ONNX_MLIR_ZHIGH_FUSION_OP_HELPER_H
