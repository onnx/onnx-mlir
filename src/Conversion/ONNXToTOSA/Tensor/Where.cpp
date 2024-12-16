// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "DialectBuilder.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class ONNXWhereLoweringToTOSA : public OpConversionPattern<ONNXWhereOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXWhereOp::Adaptor;

  LogicalResult matchAndRewrite(ONNXWhereOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value pred = adaptor.getOperands()[0];
    Value lhs = adaptor.getOperands()[1];
    Value rhs = adaptor.getOperands()[2];

    // Check types are compatible
    auto predType = dyn_cast<TensorType>(pred.getType());
    auto lhsType = dyn_cast<TensorType>(lhs.getType());
    auto rhsType = dyn_cast<TensorType>(rhs.getType());
    auto resultType = dyn_cast<TensorType>(op->getResultTypes()[0]);

    if (!predType || !lhsType || !rhsType || !resultType) {
      return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
    }
    if (!isTOSABool(predType.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "Expected bool type for condition to onnx.Where");
    }
    if (lhsType.getElementType() != rhsType.getElementType() ||
        lhsType.getElementType() != resultType.getElementType()) {
      return rewriter.notifyMatchFailure(op,
          "Expected element type for X, Y and output to be the same in "
          "onnx.Where");
    }

    // Broadcast dimensions
    IndexExprBuilderForTosa createTosaIE(rewriter, op->getLoc());
    ONNXBroadcastOpShapeHelper shapeHelper(op, {}, &createTosaIE);
    if (shapeHelper.computeShape().succeeded() &&
        shapeHelper.hasRankBroadcast()) {
      TosaBuilder tosaBuilder(rewriter, loc);
      llvm::SmallVector<Value, 4> newValues =
          tosaBuilder.equalizeRanks({pred, lhs, rhs});
      pred = newValues[0];
      lhs = newValues[1];
      rhs = newValues[2];
    }

    rewriter.replaceOpWithNewOp<mlir::tosa::SelectOp>(
        op, op.getType(), pred, lhs, rhs);
    return success();
  }
};

} // namespace

void populateLoweringONNXWhereOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXWhereLoweringToTOSA>(ctx);
}
} // namespace onnx_mlir