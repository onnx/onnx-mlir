/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Clip.cpp - Lowering Clip Op ------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Clip Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXClipOp
//===----------------------------------------------------------------------===//

struct ONNXClipOpLowering : public OpConversionPattern<ONNXClipOp> {
  // using OpConversionPattern<ONNXClipOp>::OpConversionPattern;
  ONNXClipOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXClipOp clipOp, ONNXClipOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    using LocalDialectBuilder = MultiDialectBuilder<KrnlBuilder,
        IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder>;
    Operation *op = clipOp.getOperation();
    Location loc = ONNXLoc<ONNXClipOp>(op);
    LocalDialectBuilder create(rewriter, loc);

    ValueRange operands = adaptor.getOperands();
    Value input = adaptor.getInput();
    Value min = adaptor.getMin();
    Value max = adaptor.getMax();

    // Convert the output type to MemRefType.
    Type convertedType =
        typeConverter->convertType(clipOp.getResult().getType());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();

    // Get shape.
    ONNXClipOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(input, memRefType);
    auto computeResult = [&](LocalDialectBuilder &create,
                             const ValueRange &indices) { // indices={i,j,k}
      Value loadedVal = create.krnl.load(input, indices); // load input[i,j,k]
      Value res = loadedVal;
      if (!isFromNone(min)) {
        Value minVal = create.krnl.load(min);             // load min
        Value lessThanMin = create.math.slt(res, minVal); // (input[i,j,k]<min)
        res = create.math.select(lessThanMin, minVal, res);
      }
      if (!isFromNone(max)) {
        Value maxVal = create.krnl.load(max);
        Value lessThanMax = create.math.slt(res, maxVal);
        res = create.math.select(lessThanMax, res, maxVal);
      }
      create.krnl.store(res, alloc, indices);
    };

    // Create a loop only is one of the operands is not a scalar tensor.
    if (!hasAllScalarValues(operands)) {
      uint64_t numLoops = memRefType.getRank();
      ValueRange loopDef = create.krnl.defineLoops(numLoops);

      SmallVector<IndexExpr, 4> lbs(numLoops, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      for (uint64_t i = 0; i < numLoops; ++i)
        ubs.emplace_back(shapeHelper.getOutputDims()[i]);

      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange indices) {
            LocalDialectBuilder create(createKrnl);
            computeResult(create, indices);
          });
    } else {
      computeResult(create, {});
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXClipOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXClipOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
