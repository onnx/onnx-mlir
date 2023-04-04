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
public:
  const bool USE_F32 = false;

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
    auto resMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(mmiOp.getResult().getType()));
    Type resElementType = resMemRefType.getElementType();

    // Get shape.
    ONNXMatMulIntegerOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // `vector.fma` does not support integer at this moment, and since output is
    // i32, so use f32 for computation.
    Type computeElementType = resElementType;
    if (USE_F32) {
      computeElementType = rewriter.getF32Type();
    }

    Value AInt32 = create.onnx.cast(A, computeElementType);
    Value BInt32 = create.onnx.cast(B, computeElementType);
    if (!isFromNone(aZeroPoint)) {
      Value aZeroPointInt32 = create.onnx.cast(aZeroPoint, computeElementType);
      // Fixme: using `sub` is incorrect since K is the broadcasting dim:
      // [MxK] - [M] = [MxK] - [Mx1]
      AInt32 = create.onnx.sub(AInt32, aZeroPointInt32);
    }
    if (!isFromNone(bZeroPoint)) {
      // K is the broadcating dim: [KxN] - [N] = [KxN] - [1xN]
      Value bZeroPointInt32 = create.onnx.cast(bZeroPoint, computeElementType);
      BInt32 = create.onnx.sub(BInt32, bZeroPointInt32);
    }

    Value res = create.onnx.matmul(
        RankedTensorType::get(resMemRefType.getShape(), computeElementType),
        AInt32, BInt32);
    if (USE_F32) {
      res = create.onnx.cast(res, resElementType);
    }

    rewriter.replaceOp(op, {create.onnx.toMemref(res)});
    return success();
  }
};

void populateLoweringONNXMatMulIntegerOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulIntegerOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
