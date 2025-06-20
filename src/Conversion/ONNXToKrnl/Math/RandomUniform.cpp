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
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    ArrayRef<int64_t> outputMemRefShape = outputMemRefType.getShape();
    size_t outputRank = outputMemRefShape.size();
    Type elementType = outputMemRefType.getElementType();
    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);

    // Insert alloc/dealloc pair for output tensor.
    Value alloc = create.mem.alignedAlloc(outputMemRefType);

    double high = adaptor.getHigh().convertToDouble();
    Value highValue = create.math.constant(rewriter.getF32Type(), high);
    double low = adaptor.getLow().convertToDouble();
    Value lowValue = create.math.constant(rewriter.getF32Type(), low);
    auto seed = adaptor.getSeed();
    srand(time(NULL));
    double doubleSeed = rand() % 100;
    if (seed)
      doubleSeed = seed->convertToDouble();
    Value seedValue = create.math.constant(rewriter.getF32Type(), doubleSeed);

    SmallVector<Value, 5> operands = {alloc, lowValue, highValue, seedValue};
    // Create a call to the runtime function for uniform random generation.
    rewriter.create<KrnlCallOp>(loc, "run_uniform_random", 1, operands);

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
