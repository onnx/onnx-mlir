/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- LpNormalization.cpp - Lowering LpNormalization Op --------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX LpNormalization Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#include <ctime>

using namespace mlir;

namespace onnx_mlir {

struct ONNXLpNormalizationOpLowering
    : public OpConversionPattern<ONNXLpNormalizationOp> {
public:
  ONNXLpNormalizationOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXLpNormalizationOp lpNormOp,
      ONNXLpNormalizationOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    using LocalDialectBuilder =
        MultiDialectBuilder<IndexExprBuilderForKrnl, OnnxBuilder>;
    Operation *op = lpNormOp.getOperation();
    Location loc = ONNXLoc<ONNXLpNormalizationOp>(op);
    ValueRange operands = adaptor.getOperands();
    LocalDialectBuilder create(rewriter, loc);

    // Get shape.
    ONNXLpNormalizationOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Value input = adaptor.getInput();
    auto inputType = mlir::cast<ShapedType>(input.getType());
    auto resMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(lpNormOp.getResult().getType()));
    Type elementType = resMemRefType.getElementType();

    int64_t axis = adaptor.getAxis();
    double p = adaptor.getP();
    llvm::SmallVector<int64_t> reductionShape(
        inputType.getShape().begin(), inputType.getShape().end());
    reductionShape[axis] = 1;
    llvm::SmallVector<int64_t> axesIntArray{axis};

    Value axes = create.onnx.constant(
        create.getBuilder().getI64TensorAttr(axesIntArray));
    TensorType reductionType =
        RankedTensorType::get(reductionShape, elementType);
    Value res = nullptr;
    if (p == 1) {
      // Y =  x / sum (abs(x), axis)
      Value abs = create.onnx.abs(input);
      Value sumAbs = create.onnx.reduceSum(reductionType, abs, axes);
      res = create.onnx.div(input, sumAbs);
    } else if (p == 2) {
      // Y =  x / sqrt(sum((x^2),axis))
      Value mul = create.onnx.mul(input, input);
      Value sumMul = create.onnx.reduceSum(reductionType, mul, axes);
      Value sqrtSumMul = create.onnx.sqrt(sumMul);
      res = create.onnx.div(input, sqrtSumMul);
    } else {
      llvm_unreachable(
          "The order of the normalization, only 1 or 2 are supported.");
    }

    rewriter.replaceOp(op, {create.onnx.toMemref(input)});
    return success();
  }
};

void populateLoweringONNXLpNormalizationOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLpNormalizationOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir