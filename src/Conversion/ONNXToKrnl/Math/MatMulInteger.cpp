/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- MatMulInteger.cpp - Lowering MatMulInteger Op --------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX MatMulInteger Operator to Krnl dialect.
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

struct ONNXMatMulIntegerOpLowering
    : public OpConversionPattern<ONNXMatMulIntegerOp> {
  ONNXMatMulIntegerOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXMatMulIntegerOp mmiOp,
      ONNXMatMulIntegerOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    using LocalDialectBuilder = MultiDialectBuilder<KrnlBuilder,
        IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder, OnnxBuilder>;
    Operation *op = mmiOp.getOperation();
    Location loc = ONNXLoc<ONNXMatMulIntegerOp>(op);
    LocalDialectBuilder create(rewriter, loc);

    ValueRange operands = adaptor.getOperands();
    Value A = adaptor.getA();
    Value B = adaptor.getB();
    Value aZeroPoint = mmiOp.getAZeroPoint(); // Optional input.
    Value bZeroPoint = mmiOp.getBZeroPoint(); // Optional input.

    // Common types.
    ArrayRef<int64_t> aShape = getShape(A.getType());
    ArrayRef<int64_t> bShape = getShape(B.getType());
    auto yMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(mmiOp.getResult().getType()));
    Type resElementType = yMemRefType.getElementType();

    // Get shape.
    ONNXMatMulIntegerOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Allocate output buffers.
    // Value Y =
    //    create.mem.alignedAlloc(yMemRefType, shapeHelper.getOutputDims(0));

    Value AInt32 = create.onnx.cast(A, rewriter.getF32Type());
    Value BInt32 = create.onnx.cast(B, rewriter.getF32Type());
    Value resInt32 = create.onnx.matmul(
        RankedTensorType::get(yMemRefType.getShape(), rewriter.getF32Type()),
        AInt32, BInt32);
    Value res = create.onnx.cast(resInt32, resElementType);

    rewriter.replaceOp(op, {create.onnx.toMemref(res)});
    return success();
  }
};

void populateLoweringONNXMatMulIntegerOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulIntegerOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
