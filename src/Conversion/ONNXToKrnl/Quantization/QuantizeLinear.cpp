/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- QuantizeLinear.cpp - Lowering QuantizeLinear Op -------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX QuantizeLinear Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXQuantizeLinearOpLowering
    : public OpConversionPattern<ONNXQuantizeLinearOp> {
  ONNXQuantizeLinearOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXQuantizeLinearOp qlOp,
      ONNXQuantizeLinearOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    using LocalDialectBuilder = MultiDialectBuilder<KrnlBuilder,
        IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder, OnnxBuilder>;
    Operation *op = qlOp.getOperation();
    Location loc = ONNXLoc<ONNXQuantizeLinearOp>(op);
    LocalDialectBuilder create(rewriter, loc);

    ValueRange operands = adaptor.getOperands();
    Value X = adaptor.getX();
    Value YScale = adaptor.getYScale();
    Value YZeroPoint = qlOp.getYZeroPoint(); // Optional input.

    // Only support per-tensor/layer quantization.
    MemRefType yScaleMemRefType = YScale.getType().cast<MemRefType>();
    assert(yScaleMemRefType.getRank() == 0 &&
           "Does not support per-axis quantization");

    // MemRefType for inputs and outputs.
    auto xMemRefType = dyn_cast<MemRefType>(X.getType());
    auto yMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(qlOp.getResult().getType()));

    // Types
    Type elementType = xMemRefType.getElementType();
    Type quantizedElementType = yMemRefType.getElementType();
    int64_t rank = xMemRefType.getRank();

    // Get shape.
    ONNXQuantizeLinearOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Allocate output buffers.
    Value Y =
        create.mem.alignedAlloc(yMemRefType, shapeHelper.getOutputDims(0));

    // Equations:
    // y = saturate (round (x / y_scale) + y_zero_point)
    //
    // where, saturate is to clip to [0, 255] for ui8.

    // ui8 bounds.
    Value qMax = create.math.constant(elementType, 255.0);
    Value qMin = create.math.constant(elementType, 0.0);

    // Load y_scale.
    Value scale = create.krnl.load(YScale);

    // Load y_zero_point.
    Value zeroPoint;
    if (!isFromNone(YZeroPoint)) {
      zeroPoint = create.krnl.load(adaptor.getYZeroPoint());
      zeroPoint = create.math.cast(elementType, zeroPoint);
    } else
      zeroPoint = create.math.constant(elementType, 0.0);

    // Compute y.
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr> lbs(rank, LiteralIndexExpr(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.getOutputDims(0),
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder, OnnxBuilder> create(
              createKrnl);
          Value x = create.krnl.load(X, loopInd);
          // Scale
          Value scaleX = create.math.div(x, scale);
          // Round
          Value roundX = create.onnx.round(scaleX, /*scalarType=*/true);
          // Adjust
          Value adjustX = create.math.add(roundX, zeroPoint);
          // Saturate
          Value lessThanMin = create.math.slt(adjustX, qMin);
          Value saturateX = create.math.select(lessThanMin, qMin, adjustX);
          Value lessThanMax = create.math.slt(saturateX, qMax);
          saturateX = create.math.select(lessThanMax, saturateX, qMax);
          Value res = create.math.cast(quantizedElementType, saturateX);
          create.krnl.store(res, Y, loopInd);
        });

    rewriter.replaceOp(op, {Y});
    return success();
  }
};

void populateLoweringONNXQuantizeLinearOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXQuantizeLinearOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
