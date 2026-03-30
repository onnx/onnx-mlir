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
// string (e.g., "RELU", "LEAKYRELU", "RELU6", "HSIGMOID") and preserves
// any relevant attributes (e.g., leakyrelu_alpha) from the fused activation.
//
// ReLU (α=0) is kept as "RELU" — not lowered to PRELU (differs from some
// xcompiler QDQ paths that used PRELU for ReLU).
//
// Activation normalization rules:
//
//  Input activation    | Output activation | Condition / Notes
//  --------------------|-------------------|---------------------------
//  "NONE"              | "NONE"            | No activation
//  "RELU"              | "NONE"            | If output is UINT8 and
//                      |                   | zero_point == 0 (ReLU is
//                      |                   | implicit in unsigned repr)
//  "RELU"              | "RELU"            | Otherwise: keep ReLU (no PRELU)
//  "LEAKYRELU"         | "RELU"            | If leakyrelu_alpha attr == 0
//                      |                   | (XIR convention; same as ReLU)
//  "LEAKYRELU"         | "LEAKYRELU"       | If alpha == 26/256
//  "LEAKYRELU"         | "PRELU"           | If alpha != 26/256 and != 0
//                      |                   | Computes prelu_in/shift
//  "HSIGMOID"          | "HSIGMOID"        | No change
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

/// LEAKYRELU with non-native α: set PRELU + (prelu_in, prelu_shift) from \p
/// alpha.
template <typename ConvOp>
static std::pair<int64_t, int64_t> applyPreluFixedPointForAlpha(
    ConvOp convOp, OpBuilder &builder, float alpha) {
  auto [M, N] = getPreluFactor(alpha);
  convOp.setActivationAttr(builder.getStringAttr("PRELU"));
  convOp.setPreluInAttr(getSI64Attr(builder, M));
  convOp.setPreluShiftAttr(getSI64Attr(builder, N));
  return {M, N};
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

/// ReLU (α=0) normalization: optional UINT8 implicit → NONE; otherwise keep
/// ReLU and do not lower to PRELU.
template <typename ConvOp>
static void normalizeReluActivation(
    ConvOp convOp, OpBuilder &builder, bool fromLeakyReluZeroAlpha = false) {
  // Special case: if output is UINT8 with zero_point=0, ReLU is implicit
  // in the unsigned representation — no activation needed.
  if (isReluImplicitInOutputType(convOp.getResult().getType())) {
    convOp.setActivationAttr(builder.getStringAttr("NONE"));
    LLVM_DEBUG(llvm::dbgs()
               << "NormalizeConvActivation: " << convOp->getName() << " "
               << (fromLeakyReluZeroAlpha ? "LEAKYRELU (alpha=0)" : "RELU")
               << " -> NONE (implicit in UINT8 zp=0)\n");
    return;
  }

  if (fromLeakyReluZeroAlpha) {
    // LEAKYRELU with explicit α=0 is plain ReLU — use RELU, not PRELU.
    convOp.setActivationAttr(builder.getStringAttr("RELU"));
    LLVM_DEBUG(llvm::dbgs() << "NormalizeConvActivation: " << convOp->getName()
                            << " LEAKYRELU (alpha=0) -> RELU (no PRELU)\n");
    return;
  }

  // Input is already "RELU"; leave activation and attrs unchanged (no PRELU).
}

/// Normalize the activation attribute on a conv-like op.
template <typename ConvOp>
static void normalizeActivation(ConvOp convOp, OpBuilder &builder) {
  StringRef activation = convOp.getActivation();

  // "NONE" — nothing to do
  if (activation == "NONE")
    return;

  // "RELU" handling
  if (activation == "RELU") {
    normalizeReluActivation(convOp, builder);
    return;
  }

  // "LEAKYRELU" handling
  if (activation == "LEAKYRELU") {
    // Explicit LEAKYRELU_alpha == 0 means ReLU (XIR convention): same IR as
    // RELU.
    if (auto alphaAttr = convOp.getLeakyreluAlphaAttr()) {
      if (alphaAttr.getValue().convertToFloat() == 0.0f) {
        normalizeReluActivation(
            convOp, builder, /*fromLeakyReluZeroAlpha=*/true);
        return;
      }
    }

    float alpha = kStandardLeakyReluAlpha;
    if (auto alphaAttr = convOp.getLeakyreluAlphaAttr())
      alpha = alphaAttr.getValue().convertToFloat();

    if (alpha == kStandardLeakyReluAlpha) {
      LLVM_DEBUG(llvm::dbgs()
                 << "NormalizeConvActivation: " << convOp->getName()
                 << " LEAKYRELU (alpha=26/256) -> LEAKYRELU (standard)\n");
    } else {
      auto [M, N] = applyPreluFixedPointForAlpha(convOp, builder, alpha);
      LLVM_DEBUG(llvm::dbgs()
                 << "NormalizeConvActivation: " << convOp->getName()
                 << " LEAKYRELU (alpha=" << alpha << ") -> PRELU (M=" << M
                 << ", N=" << N << ")\n");
    }
    return;
  }

  // "HSIGMOID", "RELU6", "SIGMOID" — no transformation needed, pass through.
  LLVM_DEBUG(llvm::dbgs() << "NormalizeConvActivation: " << convOp->getName()
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

  StringRef getArgument() const override { return "normalize-conv-activation"; }
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
