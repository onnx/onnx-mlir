#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

class FixNegScale : public OpRewritePattern<ONNXDequantizeLinearOp> {
public:
  using OpRewritePattern<ONNXDequantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDequantizeLinearOp dqOp, PatternRewriter &rewriter) const override {
    // Scale must be a constant.
    auto scaleOp = dqOp.getXScale().getDefiningOp<ONNXConstantOp>();
    if (!scaleOp)
      return rewriter.notifyMatchFailure(dqOp, "scale not a constant");

    auto scaleAttr =
        dyn_cast_if_present<DenseFPElementsAttr>(scaleOp.getValueAttr());
    if (!scaleAttr || scaleAttr.getNumElements() != 1)
      return rewriter.notifyMatchFailure(dqOp, "scale not a scalar float");

    APFloat scaleVal = scaleAttr.getSplatValue<APFloat>();
    if (!scaleVal.isNegative())
      return rewriter.notifyMatchFailure(dqOp, "scale is not negative");

    // Data input (x) must be a scalar integer constant with value 1.
    auto xOp = dqOp.getX().getDefiningOp<ONNXConstantOp>();
    if (!xOp)
      return rewriter.notifyMatchFailure(dqOp, "x not a constant");

    auto xAttr =
        dyn_cast_if_present<DenseIntOrFPElementsAttr>(xOp.getValueAttr());
    if (!xAttr || xAttr.getNumElements() != 1)
      return rewriter.notifyMatchFailure(dqOp, "x not a scalar");

    auto xIntType = dyn_cast<IntegerType>(xAttr.getElementType());
    if (!xIntType)
      return rewriter.notifyMatchFailure(dqOp, "x not an integer type");

    APInt xVal = xAttr.getSplatValue<APInt>();
    if (xIntType.isUnsigned() ? xVal.getZExtValue() != 1
                              : xVal.getSExtValue() != 1)
      return rewriter.notifyMatchFailure(dqOp, "x value is not 1");

    // Zero-point must be a scalar integer constant with value 0.
    auto zpOp = dqOp.getXZeroPoint().getDefiningOp<ONNXConstantOp>();
    if (!zpOp)
      return rewriter.notifyMatchFailure(dqOp, "zero point not a constant");

    auto zpAttr =
        dyn_cast_if_present<DenseIntOrFPElementsAttr>(zpOp.getValueAttr());
    if (!zpAttr || zpAttr.getNumElements() != 1)
      return rewriter.notifyMatchFailure(dqOp, "zero point not a scalar");

    auto zpIntType = dyn_cast<IntegerType>(zpAttr.getElementType());
    if (!zpIntType)
      return rewriter.notifyMatchFailure(dqOp, "zero point not integer type");

    APInt zpVal = zpAttr.getSplatValue<APInt>();
    if (zpIntType.isUnsigned() ? zpVal.getZExtValue() != 0
                               : zpVal.getSExtValue() != 0)
      return rewriter.notifyMatchFailure(dqOp, "zero point value is not 0");

    // Negate scale.
    APFloat negScale = scaleVal;
    negScale.changeSign();
    auto negScaleAttr = DenseElementsAttr::get(
        cast<ShapedType>(scaleOp.getResult().getType()), negScale);

    // Swap x and zp: new x = 0, new zp = 1.
    auto newXAttr =
        DenseElementsAttr::get(cast<ShapedType>(xOp.getResult().getType()),
            IntegerAttr::get(xIntType, 0));
    auto newZpAttr =
        DenseElementsAttr::get(cast<ShapedType>(zpOp.getResult().getType()),
            IntegerAttr::get(zpIntType, 1));

    Location loc = dqOp.getLoc();
    auto newScaleOp =
        rewriter.create<ONNXConstantOp>(loc, Attribute(), negScaleAttr);
    auto newXOp = rewriter.create<ONNXConstantOp>(loc, Attribute(), newXAttr);
    auto newZpOp = rewriter.create<ONNXConstantOp>(loc, Attribute(), newZpAttr);

    rewriter.replaceOpWithNewOp<ONNXDequantizeLinearOp>(dqOp, dqOp.getType(),
        newXOp.getResult(), newScaleOp.getResult(), newZpOp.getResult(),
        dqOp.getAxisAttr(), dqOp.getBlockSizeAttr());
    return success();
  }
};

class FixNegScalePass
    : public PassWrapper<FixNegScalePass, OperationPass<func::FuncOp>> {
  [[nodiscard]] StringRef getArgument() const override {
    return "fix-neg-scale";
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FixNegScale>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createFixNegScalePass() {
  return std::make_unique<FixNegScalePass>();
}

} // namespace onnx_mlir
