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

struct ONNXRandomNormalLikeOpLowering
    : public OpConversionPattern<ONNXRandomNormalLikeOp> {
  ONNXRandomNormalLikeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}
  LogicalResult matchAndRewrite(ONNXRandomNormalLikeOp randOp,
      ONNXRandomNormalLikeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = randOp.getOperation();
    Location loc = ONNXLoc<ONNXRandomNormalLikeOp>(op);
    Value input = adaptor.getInput();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
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
    double mean = adaptor.getMean().convertToDouble();
    Value meanValue = create.math.constant(elementType, mean);
    double scale = adaptor.getScale().convertToDouble();
    Value scaleValue = create.math.constant(elementType, scale);
    auto seed = adaptor.getSeed();
    srand(time(NULL));
    double doubleSeed = rand() % 100;
    if (seed)
      doubleSeed = seed->convertToDouble();
    Value seedValue = create.math.constant(elementType, doubleSeed);
    create.krnl.randomNormal(
        alloc, numberOfRandomValues, meanValue, scaleValue, seedValue);

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXRandomNormalLikeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRandomNormalLikeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
