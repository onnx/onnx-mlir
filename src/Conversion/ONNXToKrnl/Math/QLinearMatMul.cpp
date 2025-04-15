/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- QLinearMatMul.cpp - Lowering QLinearMatMul Op --------------===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX QLinearMatMul Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXQLinearMatMulOpLowering
    : public OpConversionPattern<ONNXQLinearMatMulOp> {
public:
  ONNXQLinearMatMulOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXQLinearMatMulOp qlmmOp,
      ONNXQLinearMatMulOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    using LocalDialectBuilder =
        MultiDialectBuilder<IndexExprBuilderForKrnl, OnnxBuilder>;
    Operation *op = qlmmOp.getOperation();
    Location loc = ONNXLoc<ONNXQLinearMatMulOp>(op);
    LocalDialectBuilder create(rewriter, loc);

    ValueRange operands = adaptor.getOperands();
    Value A = adaptor.getA();
    Value aScale = adaptor.getAScale();
    Value aZeroPoint = adaptor.getAZeroPoint();
    Value B = adaptor.getB();
    Value bScale = adaptor.getBScale();
    Value bZeroPoint = adaptor.getBZeroPoint();
    Value yScale = adaptor.getYScale();
    Value yZeroPoint = adaptor.getYZeroPoint();

    // Now only support integer8 for inputs and zeropoints, and support float32
    // for scale.
    if (!getElementType(A.getType()).isInteger(8))
      return failure();
    if (!getElementType(B.getType()).isInteger(8))
      return failure();
    if (!getElementType(aScale.getType()).isF32())
      return failure();
    if (!getElementType(bScale.getType()).isF32())
      return failure();
    if (!getElementType(yScale.getType()).isF32())
      return failure();
    if (!getElementType(aZeroPoint.getType()).isInteger(8))
      return failure();
    if (!getElementType(bZeroPoint.getType()).isInteger(8))
      return failure();
    if (!getElementType(yZeroPoint.getType()).isInteger(8))
      return failure();

    // Common types.
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();
    auto resMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(qlmmOp.getResult().getType()));
    Type resElementType = resMemRefType.getElementType();

    // Get shape.
    ONNXQLinearMatMulOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Value cst128;
    if (resElementType.isUnsignedInteger(8)) {
      auto cst128Attr = DenseElementsAttr::get(
          RankedTensorType::get({}, i32Ty), static_cast<int32_t>(128));
      cst128 = create.onnx.constant(cst128Attr);
    }

    // Prepare input A.
    Value AI8 = create.onnx.getOrCastToI8(A);
    Value AI32 = create.onnx.cast(AI8, i32Ty);
    auto aZeroPointType = mlir::cast<ShapedType>(aZeroPoint.getType());
    int64_t aZeroPointRank = aZeroPointType.getRank();
    Value aZeroPointI8 = create.onnx.getOrCastToI8(aZeroPoint);
    Value aZeroPointI32 = create.onnx.cast(aZeroPointI8, i32Ty);
    // If broadcasting, e.g. A is [MxK], zeroPoint is [M], M != 1.
    // Unsqueeze zeroPoint to [Mx1] to make shapes compatible.
    // There is no need to handle scalar zeroPoint (e.g. tensor<dtype> or
    // tensor<1xdtype>), which is always true for broadcasting.
    if ((aZeroPointRank == 1) && (aZeroPointType.getShape()[0] != 1)) {
      SmallVector<int64_t, 4> unsqueezeShape(aZeroPointType.getShape());
      unsqueezeShape.emplace_back(1);
      aZeroPointI32 =
          create.onnx.unsqueeze(RankedTensorType::get(unsqueezeShape, i32Ty),
              aZeroPointI32, create.onnx.constantInt64({aZeroPointRank}));
    }
    AI32 = create.onnx.sub(AI32, aZeroPointI32);

    // Prepare input B.
    Value BI8 = create.onnx.getOrCastToI8(B);
    Value BI32 = create.onnx.cast(BI8, i32Ty);
    // K is the broadcating dim: [KxN] - [N] = [KxN] - [1xN]
    Value bZeroPointI8 = create.onnx.getOrCastToI8(bZeroPoint);
    Value bZeroPointI32 = create.onnx.cast(bZeroPointI8, i32Ty);
    BI32 = create.onnx.sub(BI32, bZeroPointI32);

    // Prepare output Y
    Value yZeroPointI8 = create.onnx.getOrCastToI8(yZeroPoint);
    Value yZeroPointI32 = create.onnx.cast(yZeroPointI8, i32Ty);

    // Emit MatMul.
    Value resI32 = create.onnx.matmul(
        RankedTensorType::get(resMemRefType.getShape(), i32Ty), AI32, BI32);

    // Scale the output.
    Value resF32 = create.onnx.cast(resI32, f32Ty);
    Value scale = create.onnx.div(create.onnx.mul(aScale, bScale), yScale);
    resF32 = create.onnx.mul(resF32, scale);

    // Saturate and add zero point.
    Value roundToEven = create.onnx.round(resF32);
    resI32 = create.onnx.cast(roundToEven, i32Ty);
    resI32 = create.onnx.add(resI32, yZeroPointI32);
    if (resElementType.isUnsignedInteger(8)) {
      resI32 = create.onnx.add(resI32, cst128);
      resI32 = create.onnx.cast(resI32, i8Ty);
    }
    Value res = create.onnx.cast(resI32, resElementType);

    rewriter.replaceOp(op, {create.onnx.toMemref(res)});
    return success();
  }
};

void populateLoweringONNXQLinearMatMulOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXQLinearMatMulOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
