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
// For DIV (real_y = real_x / real_c) the pure-rescale equivalent is:
//   new_y_scale_onnx = s_y * real_c
// For MUL (real_y = real_x * real_c) the pure-rescale equivalent is:
//   new_y_scale_onnx = s_y / real_c
//
// where real_c = (q_c - c_zp) * c_s.  The output zero point is preserved.
// The op is replaced by an XCOMPILERRequantize whose result tensor carries
// the new (scale, zp) on its quantized element type, so that downstream
// consumers see the updated quantization parameters.
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
    static_assert(std::is_same_v<ONNXBinOp, ONNXDivOp> ||
                      std::is_same_v<ONNXBinOp, ONNXMulOp>,
        "Only Div / Mul are supported by this pattern");

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

    // The rewrite changes the result's quantized element type (new y_scale).
    // That type change propagates into every user of the op's result, so we
    // can only safely apply it when all users can absorb the new quant type
    // -- i.e. all users are ONNX ops other than Q/DQ. func.return, quant
    // ops, etc. have fixed operand types and would become ill-typed.
    auto isAllowedUser = [](Operation *user) -> bool {
      if (!user)
        return false;
      if (isa<ONNXDequantizeLinearOp, ONNXQuantizeLinearOp>(user))
        return false;
      return isa<ONNXDialect>(user->getDialect());
    };
    if (!llvm::all_of(out.getUsers(), isAllowedUser))
      return rewriter.notifyMatchFailure(binOp,
          "Output has a non-ONNX consumer; cannot update its quant "
          "type without invalidating downstream IR");

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

    // Compute new output scale so that REQUANTIZE alone reproduces the
    // integer values that DIV / MUL would have produced.
    double newYScale = outQType.getScale();
    if constexpr (std::is_same_v<ONNXBinOp, ONNXDivOp>) {
      newYScale *= realC;
    } else {
      newYScale /= realC;
    }
    newYScale = convertToExpressedType(newYScale, outQType);
    if (!std::isfinite(newYScale) || newYScale == 0.0)
      return rewriter.notifyMatchFailure(
          binOp, "Computed new y_scale is not finite / non-zero");

    int64_t newYZp = outQType.getZeroPoint();

    auto newOutQType = quant::UniformQuantizedType::getChecked(
        [&]() { return binOp->emitOpError(); }, outQType.getFlags(),
        outQType.getStorageType(), outQType.getExpressedType(), newYScale,
        newYZp, outQType.getStorageTypeMin(), outQType.getStorageTypeMax());
    if (!newOutQType)
      return rewriter.notifyMatchFailure(
          binOp, "Failed to build new output quant type");

    auto newOutType = outType.clone(newOutQType);

    // Build XCOMPILERRequantize attributes:
    //   a_scale, a_zp   = LHS quant params (unchanged)
    //   y_scale, y_zp   = new output quant params
    ArrayAttr aScaleAttr = rewriter.getArrayAttr(
        {rewriter.getF32FloatAttr(static_cast<float>(lhsQType.getScale()))});
    ArrayAttr aZpAttr = rewriter.getI64ArrayAttr({lhsQType.getZeroPoint()});
    ArrayAttr yScaleAttr = rewriter.getArrayAttr(
        {rewriter.getF32FloatAttr(static_cast<float>(newYScale))});
    ArrayAttr yZpAttr = rewriter.getI64ArrayAttr({newYZp});

    auto requantize = rewriter.create<XCOMPILERRequantizeOp>(binOp->getLoc(),
        newOutType, lhs, aScaleAttr, aZpAttr, yScaleAttr, yZpAttr);

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
