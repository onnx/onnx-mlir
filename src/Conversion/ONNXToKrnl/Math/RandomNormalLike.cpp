/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- RandomNormalLike.cpp - Lowering RandomNormal Op -------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Random Normal Like Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/MLIRDialectBuilder.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

#include <ctime>

using namespace mlir;

struct ONNXRandomNormalLikeOpLowering : public ConversionPattern {
  ONNXRandomNormalLikeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXRandomNormalLikeOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    // Get the input memref:
    ONNXRandomNormalLikeOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.input();

    // Output type:
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    ArrayRef<int64_t> outputMemRefShape = outputMemRefType.getShape();
    int outputRank = outputMemRefShape.size();
    Type elementType = outputMemRefType.getElementType();

    // Insert alloc/dealloc pair for output tensor.
    bool insertDealloc = checkInsertDealloc(op);
    Value alloc = insertAllocAndDealloc(
        outputMemRefType, loc, rewriter, insertDealloc, input);

    // Compute the number of random values required.
    int64_t constantValues = 1;
    for (decltype(outputRank) i = 0; i < outputRank; ++i)
      if (outputMemRefShape[i] > 0)
        constantValues *= outputMemRefShape[i];
    Value numberOfRandomValues =
        emitConstantOp(rewriter, loc, rewriter.getIndexType(), constantValues);

    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);

    // Incorporate any dynamic values into the number of values:
    for (decltype(outputRank) i = 0; i < outputRank; ++i) {
      if (outputMemRefShape[i] < 0) {
        Value dim = create.mem.dim(input, i);
        numberOfRandomValues = create.math.mul(numberOfRandomValues, dim);
      }
    }

    // Create the Krnl Random Normal operation:
    ONNXRandomNormalLikeOp randomNormalLikeOp =
        cast<ONNXRandomNormalLikeOp>(op);
    double mean = randomNormalLikeOp.mean().convertToDouble();
    Value meanValue = emitConstantOp(rewriter, loc, elementType, mean);
    double scale = randomNormalLikeOp.scale().convertToDouble();
    Value scaleValue = emitConstantOp(rewriter, loc, elementType, scale);
    auto seed = randomNormalLikeOp.seed();
    srand(time(NULL));
    double doubleSeed = rand() % 100;
    if (seed)
      doubleSeed = seed->convertToDouble();
    Value seedValue = emitConstantOp(rewriter, loc, elementType, doubleSeed);
    create.krnl.randomNormal(
        alloc, numberOfRandomValues, meanValue, scaleValue, seedValue);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXRandomNormalLikeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRandomNormalLikeOpLowering>(typeConverter, ctx);
}
