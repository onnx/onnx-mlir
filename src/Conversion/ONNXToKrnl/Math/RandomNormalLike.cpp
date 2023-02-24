/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- RandomNormalLike.cpp - Lowering RandomNormal Op -------------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Random Normal Like Operator to Krnl dialect.
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

struct ONNXRandomNormalLikeOpLowering : public ConversionPattern {
  ONNXRandomNormalLikeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXRandomNormalLikeOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    // Get the input memref:
    ONNXRandomNormalLikeOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.getInput();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    ArrayRef<int64_t> outputMemRefShape = outputMemRefType.getShape();
    int outputRank = outputMemRefShape.size();
    Type elementType = outputMemRefType.getElementType();
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder, MemRefBuilder>
        create(rewriter, loc);

    // Insert alloc/dealloc pair for output tensor.
    Value alloc = create.mem.alignedAlloc(input, outputMemRefType);

    // Compute the number of random values required.
    int64_t constantValues = 1;
    for (decltype(outputRank) i = 0; i < outputRank; ++i)
      if (outputMemRefShape[i] != ShapedType::kDynamic)
        constantValues *= outputMemRefShape[i];
    Value numberOfRandomValues =
        create.math.constant(rewriter.getIndexType(), constantValues);

    // Incorporate any dynamic values into the number of values:
    for (decltype(outputRank) i = 0; i < outputRank; ++i) {
      if (outputMemRefShape[i] == ShapedType::kDynamic) {
        Value dim = create.mem.dim(input, i);
        numberOfRandomValues = create.math.mul(numberOfRandomValues, dim);
      }
    }

    // Create the Krnl Random Normal operation:
    ONNXRandomNormalLikeOp randomNormalLikeOp =
        cast<ONNXRandomNormalLikeOp>(op);
    double mean = randomNormalLikeOp.getMean().convertToDouble();
    Value meanValue = create.math.constant(elementType, mean);
    double scale = randomNormalLikeOp.getScale().convertToDouble();
    Value scaleValue = create.math.constant(elementType, scale);
    auto seed = randomNormalLikeOp.getSeed();
    srand(time(NULL));
    double doubleSeed = rand() % 100;
    if (seed)
      doubleSeed = seed->convertToDouble();
    Value seedValue = create.math.constant(elementType, doubleSeed);
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

} // namespace onnx_mlir
