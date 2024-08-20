/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- RandomNormal.cpp - Lowering RandomNormal Op --------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Random Normal Operator to Krnl dialect.
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

struct ONNXRandomNormalOpLowering
    : public OpConversionPattern<ONNXRandomNormalOp> {
  ONNXRandomNormalOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}
  LogicalResult matchAndRewrite(ONNXRandomNormalOp randOp,
      ONNXRandomNormalOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = randOp.getOperation();
    Location loc = ONNXLoc<ONNXRandomNormalOp>(op);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    ArrayRef<int64_t> outputMemRefShape = outputMemRefType.getShape();
    size_t outputRank = outputMemRefShape.size();
    Type elementType = outputMemRefType.getElementType();
    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);

    // Insert alloc/dealloc pair for output tensor.
    Value alloc = create.mem.alignedAlloc(outputMemRefType);

    // Compute the number of random values required:
    int64_t randomValues = 1;
    for (decltype(outputRank) i = 0; i < outputRank; ++i)
      randomValues *= outputMemRefShape[i];
    Value numberOfRandomValues =
        create.math.constant(rewriter.getIndexType(), randomValues);

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

void populateLoweringONNXRandomNormalOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRandomNormalOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
