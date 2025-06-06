/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- RandomUniform.cpp - Lowering RandomUniform Op//----------===//
//
// Copyright 2025  The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Random Uniform Operator to Krnl dialect.
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

struct ONNXRandomUniformOpLowering
    : public OpConversionPattern<ONNXRandomUniformOp> {
  ONNXRandomUniformOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}
  LogicalResult matchAndRewrite(ONNXRandomUniformOp randOp,
      ONNXRandomUniformOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = randOp.getOperation();
    Location loc = ONNXLoc<ONNXRandomUniformOp>(op);
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

    // Create the Krnl Random Uniform operation:
    double high = adaptor.getHigh().convertToDouble();
    Value highValue = create.math.constant(elementType, high);
    double low = adaptor.getLow().convertToDouble();
    Value lowValue = create.math.constant(elementType, low);
    auto seed = adaptor.getSeed();
    srand(time(NULL));
    double doubleSeed = rand() % 100;
    if (seed)
      doubleSeed = seed->convertToDouble();
    Value seedValue = create.math.constant(elementType, doubleSeed);

    create.krnl.randomUniform(
        alloc, numberOfRandomValues, lowValue, highValue, seedValue);

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXRandomUniformOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRandomUniformOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
