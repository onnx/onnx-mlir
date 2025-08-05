
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
using namespace mlir;
using namespace onnx_mlir;
namespace {

float getFloatFromConstant(Value val) {
  auto constOp = val.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return 0.0f;
  auto attr = constOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  if (!attr || attr.getNumElements() != 1)
    return 0.0f;
  auto floatAttr = (*attr.getValues<FloatAttr>().begin());
  return floatAttr.getValueAsDouble();
}

int64_t getIntFromConstant(Value val) {
  auto constOp = val.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return 0;
  auto attr = constOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  if (!attr || attr.getNumElements() != 1)
    return 0;
  auto intAttr = (*attr.getValues<IntegerAttr>().begin());
  return intAttr.getInt();
}

bool quantizationParamsMatch(
    Value scale1, Value zp1, Value scale2, Value zp2, float tolerance = 1e-5f) {
  float s1 = getFloatFromConstant(scale1);
  float s2 = getFloatFromConstant(scale2);
  int64_t z1 = getIntFromConstant(zp1);
  int64_t z2 = getIntFromConstant(zp2);
  bool zeroPointMatch = (z1 == z2);
  bool scaleClose = std::fabs(s1 - s2) < tolerance;
  return zeroPointMatch && scaleClose;
}

struct RemoveDQConcatQPattern : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {
    auto concatOp = qOp.getX().getDefiningOp<ONNXConcatOp>();
    if (!concatOp) {
      return failure();
    }
    if (concatOp.getInputs().size() != 1) {
      return failure();
    }
    auto dqOp = concatOp.getInputs()[0].getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp) {
      return failure();
    }
    if (!quantizationParamsMatch(dqOp.getXScale(), dqOp.getXZeroPoint(),
            qOp.getYScale(), qOp.getYZeroPoint())) {
      return failure();
    }
    rewriter.replaceOp(qOp, dqOp.getX());
    return success();
  }
};

struct ConcatOptONNXToONNXPass
    : public PassWrapper<ConcatOptONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConcatOptONNXToONNXPass)
  StringRef getArgument() const override { return "concat-opt-onnx-to-onnx"; }
  StringRef getDescription() const override {
    return "Optimize concat operations with quantization in ONNX dialect";
  }
  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    // llvm::outs() << " ****************** 2. CreateConcCat **************"
    //              << "\n";
    patterns.add<RemoveDQConcatQPattern>(&getContext());
    if (failed(applyPatternsGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createConcatOptONNXToONNXPass() {
  // llvm::outs() << " ****************** 1. CreateConcCat **************" <<
  // "\n";
  return std::make_unique<ConcatOptONNXToONNXPass>();
}

} // namespace onnx_mlir