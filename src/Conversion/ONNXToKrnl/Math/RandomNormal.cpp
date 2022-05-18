/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- RandomNormal.cpp - Lowering RandomNormal Op --------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

#include <ctime>

using namespace mlir;

namespace onnx_mlir {

struct ONNXRandomNormalOpLowering : public ConversionPattern {
  ONNXRandomNormalOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXRandomNormalOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();

    ArrayRef<int64_t> outputMemRefShape = outputMemRefType.getShape();
    size_t outputRank = outputMemRefShape.size();
    Type elementType = outputMemRefType.getElementType();

    // Insert alloc/dealloc pair for output tensor.
    bool insertDealloc = checkInsertDealloc(op);
    Value alloc =
        insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);

    // Compute the number of random values required:
    int64_t randomValues = 1;
    for (decltype(outputRank) i = 0; i < outputRank; ++i)
      randomValues *= outputMemRefShape[i];
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
    Value numberOfRandomValues =
        create.math.constant(rewriter.getIndexType(), randomValues);

    // Create the Krnl Random Normal operation:
    ONNXRandomNormalOp randomNormalOp = llvm::cast<ONNXRandomNormalOp>(op);
    double mean = randomNormalOp.mean().convertToDouble();
    Value meanValue = create.math.constant(elementType, mean);
    double scale = randomNormalOp.scale().convertToDouble();
    Value scaleValue = create.math.constant(elementType, scale);
    auto seed = randomNormalOp.seed();
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

void populateLoweringONNXRandomNormalOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRandomNormalOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
