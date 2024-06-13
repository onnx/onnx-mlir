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
    assert(resMemRefType.getElementType() == rewriter.getI32Type() &&
           "Output element type must be i32");

    // Get shape.
    ONNXMatMulIntegerOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Prepare input A.
    Value AInt32 = create.onnx.cast(A, resElementType);
    if (!isNoneValue(aZeroPoint)) {
      auto aZeroPointType = mlir::cast<ShapedType>(aZeroPoint.getType());
      int64_t aZeroPointRank = aZeroPointType.getRank();
      Value aZeroPointInt32 = create.onnx.cast(aZeroPoint, resElementType);
      // If broadcasting, e.g. A is [MxK], zeroPoint is [M], M != 1.
      // Unsqueeze zeroPoint to [Mx1] to make shapes compatible.
      // There is no need to handle scalar zeroPoint (e.g. tensor<dtype> or
      // tensor<1xdtype>), which is always true for broadcasting.
      if ((aZeroPointRank == 1) && (aZeroPointType.getShape()[0] != 1)) {
        SmallVector<int64_t, 4> unsqueezeShape(aZeroPointType.getShape());
        unsqueezeShape.emplace_back(1);
        aZeroPointInt32 = create.onnx.unsqueeze(
            RankedTensorType::get(unsqueezeShape, resElementType),
            aZeroPointInt32, create.onnx.constantInt64({aZeroPointRank}));
      }
      AInt32 = create.onnx.sub(AInt32, aZeroPointInt32);
    }

    // Prepare input B.
    Value BInt32 = create.onnx.cast(B, resElementType);
    if (!isNoneValue(bZeroPoint)) {
      // K is the broadcating dim: [KxN] - [N] = [KxN] - [1xN]
      Value bZeroPointInt32 = create.onnx.cast(bZeroPoint, resElementType);
      BInt32 = create.onnx.sub(BInt32, bZeroPointInt32);
    }

    // Emit MatMul.
    Value res = create.onnx.matmul(
        RankedTensorType::get(resMemRefType.getShape(), resElementType), AInt32,
        BInt32);
    rewriter.replaceOp(op, {create.onnx.toMemref(res)});
    return success();
  }
};

void populateLoweringONNXMatMulIntegerOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulIntegerOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
