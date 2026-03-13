/*
 *  © Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
 */

//===- NormalizeConvActivationPass.cpp - Normalize Conv Activation Attr ---===//
//
// This pass normalizes the "activation" attribute on convolution ops
// (XFEConv, XFEConvTranspose, XCOMPILERDepthwiseConv) into the final
// hardware-compatible form, mirroring the activation handling logic from
// the xcompiler ReplaceQDQConvPass.
//
// It runs after FuseConvActivationPass, which sets the high-level activation
// string (e.g., "RELU", "LEAKYRELU", "RELU6", "HARDSIGMOID") and preserves
// any relevant attributes (e.g., leakyrelu_alpha) from the fused activation.
//
// Activation normalization rules (matching xcompiler behavior):
//
//  Input activation    | Output activation | Condition / Notes
//  --------------------|-------------------|---------------------------
//  "NONE"              | "NONE"            | No activation
//  "RELU"              | "NONE"            | If output is UINT8 and
//                      |                   | zero_point == 0 (ReLU is
//                      |                   | implicit in unsigned repr)
//  "RELU"              | "LEAKYRELU"       | Otherwise: treated as
//                      |                   | LeakyReLU with alpha=0
//                      |                   | Sets leakyrelu_alpha=0.0
//                      |                   | Then alpha(0)!=26/256 so
//                      |                   | becomes "PRELU" with
//                      |                   | computed prelu_in/shift
//  "LEAKYRELU"         | "LEAKYRELU"       | If alpha == 26/256
//  "LEAKYRELU"         | "PRELU"           | If alpha != 26/256
//                      |                   | Computes prelu_in/shift
//  "HARDSIGMOID"       | "HSIGMOID"        | Renamed for hardware
//  "RELU6"             | "RELU6"           | No change
//  "SIGMOID"           | "SIGMOID"         | No change
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "normalize-conv-activation"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// The standard LeakyReLU alpha for hardware (26/256 ≈ 0.1015625).
/// When alpha matches this value exactly, the hardware supports it natively
/// as "LEAKYRELU". Any other alpha requires the PRELU fixed-point path.
static constexpr float kStandardLeakyReluAlpha = 26.0f / 256.0f;

/// Convert LeakyReLU alpha to fixed-point PRELU representation.
/// Returns (M, N) where alpha ≈ M / 2^N.
/// This matches xcompiler's get_leakyrelu_alpha_to_prelu_factor().
static std::pair<int64_t, int64_t> getPreluFactor(float alpha) {
  int64_t N = 8;
  int64_t M = static_cast<int64_t>(std::llround(std::exp2(N) * alpha));
  return {M, N};
}

/// Create a signed i64 integer attribute (si64).
static IntegerAttr getSI64Attr(OpBuilder &builder, int64_t value) {
  auto si64 = IntegerType::get(
      builder.getContext(), 64, IntegerType::SignednessSemantics::Signed);
  return builder.getIntegerAttr(si64, value);
}

/// Check if the output type is unsigned 8-bit quantized with zero_point == 0.
/// When this is true, ReLU is implicit (unsigned representation cannot
/// represent negative values when zp=0), so no explicit activation is needed.
static bool isReluImplicitInOutputType(Type outputType) {
  auto tensorType = dyn_cast<RankedTensorType>(outputType);
  if (!tensorType)
    return false;

  auto quantType =
      dyn_cast<quant::UniformQuantizedType>(tensorType.getElementType());
  if (!quantType)
    return false;

  // Check: unsigned 8-bit storage and zero_point == 0
  return !quantType.isSigned() &&
         quantType.getStorageTypeIntegralWidth() == 8 &&
         quantType.getZeroPoint() == 0;
}

/// Normalize the activation attribute on a conv-like op.
/// This implements the xcompiler ReplaceQDQConvPass activation handling logic.
template <typename ConvOp>
static void normalizeActivation(ConvOp convOp, OpBuilder &builder) {
  StringRef activation = convOp.getActivation();

  // "NONE" — nothing to do
  if (activation == "NONE")
    return;

  // "RELU" handling
  if (activation == "RELU") {
    // Special case: if output is UINT8 with zero_point=0, ReLU is implicit
    // in the unsigned representation — no activation needed.
    if (isReluImplicitInOutputType(convOp.getResult().getType())) {
      convOp->setAttr("activation", builder.getStringAttr("NONE"));
      LLVM_DEBUG(llvm::dbgs()
                 << "NormalizeConvActivation: " << convOp->getName()
                 << " RELU -> NONE (implicit in UINT8 zp=0)\n");
      return;
    }

    // Standard case: RELU is treated as LEAKYRELU with alpha=0.
    // Since alpha=0 != 26/256, this becomes PRELU with computed mul/shift.
    float alpha = 0.0f;
    auto [M, N] = getPreluFactor(alpha);
    convOp->setAttr("activation", builder.getStringAttr("PRELU"));
    convOp->setAttr(
        "leakyrelu_alpha", builder.getF32FloatAttr(alpha));
    convOp->setAttr("prelu_in", getSI64Attr(builder, M));
    convOp->setAttr("prelu_shift", getSI64Attr(builder, N));

    LLVM_DEBUG(llvm::dbgs()
               << "NormalizeConvActivation: " << convOp->getName()
               << " RELU -> PRELU (alpha=0, M=" << M << ", N=" << N << ")\n");
    return;
  }

  // "LEAKYRELU" handling
  if (activation == "LEAKYRELU") {
    // Get alpha — it was preserved by FuseConvActivationPass from the
    // FusedEltwise op. If not present, default to the standard value.
    float alpha = kStandardLeakyReluAlpha;
    if (auto alphaAttr = convOp->template getAttrOfType<FloatAttr>(
            "leakyrelu_alpha")) {
      alpha = alphaAttr.getValue().convertToFloat();
    }

    if (alpha == kStandardLeakyReluAlpha) {
      // Standard LeakyReLU alpha — hardware supports natively.
      // Keep activation as "LEAKYRELU".
      LLVM_DEBUG(llvm::dbgs()
                 << "NormalizeConvActivation: " << convOp->getName()
                 << " LEAKYRELU (alpha=26/256) -> LEAKYRELU (standard)\n");
    } else {
      // Non-standard alpha — convert to PRELU with fixed-point mul/shift.
      auto [M, N] = getPreluFactor(alpha);
      convOp->setAttr("activation", builder.getStringAttr("PRELU"));
      convOp->setAttr("prelu_in", getSI64Attr(builder, M));
      convOp->setAttr("prelu_shift", getSI64Attr(builder, N));

      LLVM_DEBUG(llvm::dbgs()
                 << "NormalizeConvActivation: " << convOp->getName()
                 << " LEAKYRELU (alpha=" << alpha << ") -> PRELU (M=" << M
                 << ", N=" << N << ")\n");
    }
    return;
  }

  // "HARDSIGMOID" → "HSIGMOID" (rename for hardware compatibility)
  if (activation == "HARDSIGMOID") {
    convOp->setAttr("activation", builder.getStringAttr("HSIGMOID"));
    LLVM_DEBUG(llvm::dbgs()
               << "NormalizeConvActivation: " << convOp->getName()
               << " HARDSIGMOID -> HSIGMOID\n");
    return;
  }

  // "RELU6", "SIGMOID" — no transformation needed, pass through.
  LLVM_DEBUG(llvm::dbgs()
             << "NormalizeConvActivation: " << convOp->getName()
             << " activation=" << activation << " (no change)\n");
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct NormalizeConvActivationPass
    : public PassWrapper<NormalizeConvActivationPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeConvActivationPass)

  NormalizeConvActivationPass() = default;
  NormalizeConvActivationPass(const NormalizeConvActivationPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const override {
    return "normalize-conv-activation";
  }
  StringRef getDescription() const override {
    return "Normalize conv activation attributes into hardware-compatible "
           "form (LEAKYRELU/PRELU/HSIGMOID) matching xcompiler behavior";
  }

  void runOnOperation() override {
    auto function = getOperation();
    OpBuilder builder(&getContext());

    function.walk([&](Operation *op) {
      if (auto convOp = dyn_cast<XFEConvOp>(op)) {
        normalizeActivation(convOp, builder);
      } else if (auto convOp = dyn_cast<XFEConvTransposeOp>(op)) {
        normalizeActivation(convOp, builder);
      } else if (auto convOp = dyn_cast<XCOMPILERDepthwiseConvOp>(op)) {
        normalizeActivation(convOp, builder);
      }
    });
  }
};

} // end anonymous namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createNormalizeConvActivationPass() {
  return std::make_unique<NormalizeConvActivationPass>();
}
} // namespace onnx_mlir
