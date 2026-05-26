// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//
// TransferScalarConstInputDivToRequantizePass
//
// XMC pass that converts quantized DIV / MUL by a scalar quantized constant
// into a single XCOMPILERRequantize op. This is the MLIR-level analogue of
// xcompiler's TransferScalarConstInputDivToRequantizePass.
//
// At this stage (after QuantTypesPass), the IR looks like:
//
//   %const_q = onnx.Constant ... : tensor<1x!quant.uniform<ui16:f32, c_s,
//   c_zp>> %y_q     = onnx.Div %x_q, %const_q
//        : tensor<...,!quant.uniform<...,s_x,zp_x>>,
//          tensor<1x!quant.uniform<ui16:f32,c_s,c_zp>>
//          -> tensor<...,!quant.uniform<...,s_y,zp_y>>
//
// For DIV (real_y = real_x / real_c) the equivalent REQUANTIZE kernel scale
// (i.e. the y_scale used inside the rescale arithmetic) is:
//   kernel_y_scale = s_y * real_c
// For MUL (real_y = real_x * real_c) it is:
//   kernel_y_scale = s_y / real_c
//
// where real_c = (q_c - c_zp) * c_s.  The output zero point is preserved.
//
// IMPORTANT: only the REQUANTIZE op's `y_scale` *attribute* takes the new
// kernel scale.  The result tensor's quantized element type retains the
// ORIGINAL (s_y, zp_y) of the Div / Mul output, so downstream consumers see
// the same advertised scale they would have seen for the original Div / Mul.
// The integer values produced by the new REQUANTIZE with kernel scale
// (s_y * real_c) are exactly the integers that DIV would have produced and,
// when interpreted with the advertised scale s_y, represent real_x / real_c.
//
// Only scalar (single-element) quantized constants are matched, matching
// the xcompiler template filter that limits this to per-tensor constants.
//===----------------------------------------------------------------------===//

#include <cmath>
#include <optional>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/// Return the dequantized real value (q - zp) * scale of a scalar quantized
/// ONNX constant whose element type is quant::UniformQuantizedType.
/// Returns std::nullopt if rhs is not a scalar quantized constant.
std::optional<double> getScalarQuantConst(Value rhs) {
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!rhsType)
    return std::nullopt;

  auto qType = dyn_cast<quant::UniformQuantizedType>(rhsType.getElementType());
  if (!qType)
    return std::nullopt;

  auto constOp = rhs.getDefiningOp<ONNXConstantOp>();
  if (!constOp || !constOp.getValueAttr())
    return std::nullopt;

  auto attr = dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
  if (!attr || !attr.isSplat())
    return std::nullopt;

  // Match xcompiler's "1 == new_element_num" filter: scalar only.
  if (rhsType.hasStaticShape() && rhsType.getNumElements() != 1)
    return std::nullopt;

  // Storage element type must match the underlying attribute element type.
  if (qType.getStorageType() != attr.getElementType())
    return std::nullopt;

  int64_t constVal;
  if (attr.getElementType().isUnsignedInteger())
    constVal = static_cast<int64_t>(attr.getSplatValue<APInt>().getZExtValue());
  else
    constVal = attr.getSplatValue<APInt>().getSExtValue();

  return static_cast<double>(constVal - qType.getZeroPoint()) *
         qType.getScale();
}

/// Convert a value to be representable in the quant type's expressed type
/// (typically f32) so subsequent comparisons / attributes use the same
/// precision as the runtime kernel.
double convertToExpressedType(double value, quant::UniformQuantizedType qType) {
  if (auto fltType = dyn_cast<FloatType>(qType.getExpressedType())) {
    APFloat ap(value);
    bool losesInfo;
    ap.convert(
        fltType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
    value = ap.convertToDouble();
  }
  return value;
}

/// Pattern: Div/Mul by a scalar quantized constant -> XCOMPILERRequantize.
template <typename ONNXBinOp>
class TransferScalarConstInputBinToRequantizePattern
    : public OpRewritePattern<ONNXBinOp> {
public:
  using OpRewritePattern<ONNXBinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXBinOp binOp, PatternRewriter &rewriter) const override {
    if constexpr (!std::is_same_v<ONNXBinOp, ONNXDivOp> &&
                  !std::is_same_v<ONNXBinOp, ONNXMulOp>)
      return rewriter.notifyMatchFailure(
          binOp, "Only Div / Mul are supported by this pattern");

    Value lhs = binOp->getOperand(0);
    Value rhs = binOp->getOperand(1);
    Value out = binOp->getResult(0);

    // xcompiler explicitly skips the case where the *first* input is the
    // constant; canonicalization should already place constants on the RHS.
    if (lhs.template getDefiningOp<ONNXConstantOp>())
      return rewriter.notifyMatchFailure(binOp, "LHS must not be a constant");

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto outType = dyn_cast<RankedTensorType>(out.getType());
    if (!lhsType || !outType)
      return rewriter.notifyMatchFailure(binOp, "Operands not ranked tensors");

    auto lhsQType =
        dyn_cast<quant::UniformQuantizedType>(lhsType.getElementType());
    auto outQType =
        dyn_cast<quant::UniformQuantizedType>(outType.getElementType());
    if (!lhsQType || !outQType)
      return rewriter.notifyMatchFailure(
          binOp, "Input / output element types are not per-tensor quantized");

    // Match xcompiler template: only UINT16 const-fix is supported.
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!rhsType)
      return rewriter.notifyMatchFailure(binOp, "RHS not a ranked tensor");
    auto rhsQType =
        dyn_cast<quant::UniformQuantizedType>(rhsType.getElementType());
    if (!rhsQType)
      return rewriter.notifyMatchFailure(
          binOp, "RHS element type is not per-tensor quantized");
    auto rhsStorage = dyn_cast<IntegerType>(rhsQType.getStorageType());
    if (!rhsStorage || !rhsStorage.isUnsigned() || rhsStorage.getWidth() != 16)
      return rewriter.notifyMatchFailure(
          binOp, "RHS storage type must be UINT16");

    auto rhsRealOpt = getScalarQuantConst(rhs);
    if (!rhsRealOpt)
      return rewriter.notifyMatchFailure(
          binOp, "RHS is not a scalar quantized constant");
    double realC = *rhsRealOpt;

    // Skip if dequantized const is zero (matches xcompiler guard which
    // bails out with a warning rather than producing a divide-by-zero).
    if (realC == 0.0)
      return rewriter.notifyMatchFailure(
          binOp, "Dequantized scalar constant is zero");

    // Compute the kernel y_scale that the REQUANTIZE math must use so that
    // it reproduces the integer values that DIV / MUL would have produced.
    // The advertised tensor type retains the original output scale, so the
    // consumers continue to see (s_y, zp_y).
    double kernelYScale = outQType.getScale();
    if constexpr (std::is_same_v<ONNXBinOp, ONNXDivOp>) {
      kernelYScale *= realC;
    } else {
      kernelYScale /= realC;
    }
    kernelYScale = convertToExpressedType(kernelYScale, outQType);
    if (!std::isfinite(kernelYScale) || kernelYScale == 0.0)
      return rewriter.notifyMatchFailure(
          binOp, "Computed kernel y_scale is not finite / non-zero");

    int64_t yZp = outQType.getZeroPoint();

    // Build XCOMPILERRequantize attributes:
    //   a_scale, a_zp   = LHS quant params (unchanged)
    //   y_scale         = kernel scale (s_y * real_c for Div, s_y / real_c
    //                     for Mul) used by the rescale kernel
    //   y_zp            = original output zero point
    ArrayAttr aScaleAttr = rewriter.getArrayAttr(
        {rewriter.getF32FloatAttr(static_cast<float>(lhsQType.getScale()))});
    ArrayAttr aZpAttr = rewriter.getI64ArrayAttr({lhsQType.getZeroPoint()});
    ArrayAttr yScaleAttr = rewriter.getArrayAttr(
        {rewriter.getF32FloatAttr(static_cast<float>(kernelYScale))});
    ArrayAttr yZpAttr = rewriter.getI64ArrayAttr({yZp});

    // Keep the result tensor type identical to the original Div / Mul output,
    // so downstream ops continue to see (s_y, zp_y) on their input.
    auto requantize = rewriter.create<XCOMPILERRequantizeOp>(binOp->getLoc(),
        outType, lhs, aScaleAttr, aZpAttr, yScaleAttr, yZpAttr);

    rewriter.replaceOp(binOp, requantize.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct TransferScalarConstInputDivToRequantizePass
    : public PassWrapper<TransferScalarConstInputDivToRequantizePass,
          OperationPass<func::FuncOp>> {
  [[nodiscard]] StringRef getArgument() const override {
    return "transfer-scalar-const-input-div-to-requantize";
  }
  [[nodiscard]] StringRef getDescription() const override {
    return "Convert quantized Div / Mul by a scalar quantized constant into "
           "an XCOMPILERRequantize op (XMC analogue of xcompiler's "
           "TransferScalarConstInputDivToRequantizePass).";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TransferScalarConstInputBinToRequantizePattern<ONNXDivOp>,
        TransferScalarConstInputBinToRequantizePattern<ONNXMulOp>>(ctx);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass>
createTransferScalarConstInputDivToRequantizePass() {
  return std::make_unique<TransferScalarConstInputDivToRequantizePass>();
}

} // namespace onnx_mlir
