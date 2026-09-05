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
  /// See the kMaxOpCount contract note in FusionOpHelper.hpp: initial LT +
  /// split-reshape + transpose + merge-reshape + final-LT/dlf16.
  static constexpr int kMaxOpCount = 5;

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
  /// See the kMaxOpCount contract note in FusionOpHelper.hpp: unsqueeze +
  /// expand + mul + reshape + stick.
  static constexpr int kMaxOpCount = 5;

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

//===----------------------------------------------------------------------===//
// ConcatExpandStickFusionHelper
//
// Subclass for ONNXFusedOp(kind = "zhigh.concat-expand-stick").
//
// Pattern (the GQA/MQA "repeat KV heads after cache-concat" idiom):
//  ONNXConcatOp     exactly 2 inputs; axis A is not the innermost dim; each
//                   input's innermost dim static mod 64 (required)
//  ONNXUnsqueezeOp  one axis P, normalized and strictly less than the
//                   concat result's rank (so the new dim is inserted at or
//                   before the concat axis, never appended after it)
//                   (required)
//  ZHighF32ToDLF16Op  F32 CPU tensor => DLF16 CPU tensor (optional; present
//                   <=> the existing 2-step tail, ending in
//                   ONNXLayoutTransformOp; absent <=> the 1-step tail,
//                   ending in an explicit ZHighStickOp -- tracked by the
//                   local hasSingleStick in detectIfBeneficial)
//  ONNXExpandOp     dim P expands from 1 to N (N static, >= 2); every other
//                   dim is provably unchanged (required); identical check
//                   regardless of tail
//  ONNXMulOp        element-wise mul by scalar F32/I32/I64 const (optional;
//                   when absent, mulScalar stays at its neutral 1.f default)
//                   -- stick-tail only: by this point in the 2-step tail,
//                   data is already DLF16, so a real Mul there would need an
//                   F16-typed scalar (ONNX's Mul requires both operands to
//                   share one element type), which the F32/I32/I64-only
//                   scalar extraction can never match -- not tried there
//  ONNXReshapeOp    dims 0..P may collapse; dims after P unchanged (required);
//                   identical check regardless of tail
// -- then either --
//  ONNXLayoutTransformOp   CPU => ZTensor, target layout 3D / 3DS / 4D
//                   (required when F32ToDLF16 was matched); sets finalLayout
// -- or --
//  ZHighStickOp     stick to 3D / 3DS / 4D (required when F32ToDLF16 was not
//                   matched); sets stickFormat
//
// Exactly one of finalLayout / stickFormat is set after a successful match.
//
// Unique-use invariant: every intermediate value from the Unsqueeze through
// the Reshape has exactly one use.  The final result (LayoutTransform's or
// the Stick's output, depending on the tail) is not checked.  The concat
// result is the one exception: it is allowed extra uses beyond the
// Unsqueeze that starts the rest of the chain (e.g. it may also be the
// KV-cache "present" output threaded to the next layer).  When such extra
// uses exist, the ONNXFusedOp gets a second result -- the concat result
// itself -- so those uses keep a value to bind to once the chain ops move
// into the FusedOp body; see yieldConcatResult below.  Output 0 is always
// the primary result.
//===----------------------------------------------------------------------===//

class ConcatExpandStickFusionHelper : public onnx_mlir::FusionOpKindHelper {
public:
  static constexpr llvm::StringLiteral kKind{"zhigh.concat-expand-stick"};
  /// See the kMaxOpCount contract note in FusionOpHelper.hpp: concat +
  /// unsqueeze + expand + mul + reshape + stick (this kind's longest chain,
  /// tying the existing concat+unsqueeze+f32dlf16+expand+reshape+LT chain --
  /// Mul is stick-tail-only, see the class doc comment above).
  /// Deliberately larger than ExpandMulStickFusionHelper::kMaxOpCount (5):
  /// this kind's chain can subsume that one's entirely (Concat heading the
  /// same Unsqueeze->Expand->Mul->Reshape->Stick tail), so
  /// FusionOpStickUnstick.cpp gives it its own, earlier pattern-application
  /// phase -- see that file for why PatternBenefit alone can't express this.
  static constexpr int kMaxOpCount = 6;

  int64_t concatAxis = -1;         ///< A: onnx.Concat's axis (normalized)
  int64_t unsqueezedPosition = -1; ///< P: axis inserted by unsqueeze
  int64_t expansionN = -1;         ///< N: value dim P expands to
  bool noSaturation = false;       ///< F32ToDLF16's no_saturation attr
  int64_t reshapeFirstCollapsedDim =
      -1; ///< first input dim in merge run (-1 = none)
  int64_t reshapeCollapsedCount = 0; ///< # consecutive input dims merged into 1
  /// Target layout of the trailing ONNXLayoutTransformOp, for the existing
  /// 2-step tail (F32ToDLF16 -> Expand -> Reshape -> LayoutTransform). Set
  /// exactly when `stickFormat` below is not.
  std::optional<mlir::StringAttr> finalLayout; ///< "3D", "3DS", or "4D"
  bool yieldConcatResult = false; ///< concat result used outside the chain
                                  ///< too => it becomes FusedOp output 1
  /// Scalar multiplier (F32; 1 = neutral) of the optional Mul between Expand
  /// and Reshape, in the 1-step stick tail only (see the class doc comment
  /// for why the 2-step tail can never actually have one). Stays at its
  /// neutral default for the 2-step tail, and for the stick tail when no
  /// Mul op is present.
  float mulScalar = 1.f;
  /// Target layout ("3D"/"3DS"/"4D") of the trailing ZHighStickOp, for the
  /// 1-step tail (Expand -> Mul? -> Reshape -> ZHighStickOp). Set exactly
  /// when `finalLayout` above is not -- exactly one of the two is present
  /// after a successful match.
  std::optional<mlir::StringAttr> stickFormat;

  /// Detect and parameterize the concat-expand-stick chain.
  /// \p dimAnalysis must be non-null.
  bool detectIfBeneficial(
      const DimAnalysis *dimAnalysis, mlir::ONNXConcatOp startOp);

  llvm::StringRef getKind() const override { return kKind; }
  void embedAttrs(mlir::ONNXFusedOp fusedOp) const override;
  bool retrieveAttrs(mlir::ONNXFusedOp fusedOp) override;
  bool verify() const override;
};

} // namespace zhigh
} // namespace onnx_mlir

#endif // ONNX_MLIR_ZHIGH_FUSION_OP_HELPER_H
