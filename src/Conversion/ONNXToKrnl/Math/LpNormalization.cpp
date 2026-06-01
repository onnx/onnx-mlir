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
    int64_t rank = inputType.getRank();

    int64_t axis = adaptor.getAxis();
    if (axis < 0)
      axis += rank;
    assert(axis >= 0 && axis < rank && "out of bound axis");
    double p = adaptor.getP();
    llvm::SmallVector<int64_t> reductionShape(
        inputType.getShape().begin(), inputType.getShape().end());
    reductionShape[axis] = 1;
    llvm::SmallVector<int64_t> axesIntArray{axis};

    Value axes = create.onnx.constant(
        create.getBuilder().getI64TensorAttr(axesIntArray));
    TensorType reductionType =
        RankedTensorType::get(reductionShape, elementType);

#if 1
    // TODO: if the performance of this op matters, we should introduce an
    // efficient divide by 0 results in zero operator. Also, the ReduceL1 &
    // ReduceL2 ops should be natively implemented; currently the onnx ops are
    // decomposed in element operations.
    Value divisor;
    if (p == 1.0) {
      Value abs = create.onnx.abs(input);
      divisor = create.onnx.reduceSum(reductionType, abs, axes);
    } else if (p == 2.0) {
      Value mul = create.onnx.mul(input, input);
      Value sumMul = create.onnx.reduceSum(reductionType, mul, axes);
      divisor = create.onnx.sqrt(sumMul);
    } else {
      llvm_unreachable(
          "The order of the normalization, only 1 or 2 are supported.");
    }
    RankedTensorType scalarType = RankedTensorType::get({}, elementType);
    Value zero = create.onnx.constant(
        DenseElementsAttr::get(scalarType, static_cast<float>(0.0)));
    Value one = create.onnx.constant(
        DenseElementsAttr::get(scalarType, static_cast<float>(1.0)));
    Value isZero = create.onnx.equal(divisor, zero);
    // When divisor is zero, then saveDividend = 0, saveDivisor = 1.0,
    // and we have 0/1 = 0. Otherwise we have the normal input / divisor.
    Value saveDividend = create.onnx.where(inputType, isZero, zero, input);
    Value safeDivisor = create.onnx.where(reductionType, isZero, one, divisor);
    Value res = create.onnx.div(saveDividend, safeDivisor);
#else
    // TODO: the algorithm below does not follow the specs in the aspect listed
    // below. The output is computed as: output = input / Lp_norm(input, axis).
    // When the Lp norm is zero (i.e., all elements along the axis are zero),
    // the output is defined to be zero to avoid division by zero.

    if (p == 1) {
      // Y =  x / (sum(abs(x), axis) + eps)
      Value abs = create.onnx.abs(input);
      Value sumAbs = create.onnx.reduceSum(reductionType, abs, axes);
      res = create.onnx.div(input, sumAbs);
    } else if (p == 2) {
      // Y =  x / (sqrt(sum(x^2, axis)) + eps)
      Value mul = create.onnx.mul(input, input);
      Value sumMul = create.onnx.reduceSum(reductionType, mul, axes);
      Value sqrtSumMul = create.onnx.sqrt(sumMul);
      res = create.onnx.div(input, sqrtSumMul);
    } else {
      llvm_unreachable(
          "The order of the normalization, only 1 or 2 are supported.");
    }

#endif

    rewriter.replaceOp(op, res);
    return success();
  }
};

void populateLoweringONNXLpNormalizationOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLpNormalizationOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir