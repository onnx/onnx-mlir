/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- LpNormalization.cpp - Lowering LpNormalization Op --------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX LpNormalization Operator to Krnl dialect.
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

struct ONNXLpNormalizationOpLowering
    : public OpConversionPattern<ONNXLpNormalizationOp> {
public:
  ONNXLpNormalizationOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXLpNormalizationOp lpNormOp,
      ONNXLpNormalizationOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    Operation *op = lpNormOp.getOperation();
    Location loc = ONNXLoc<ONNXLpNormalizationOp>(op);
    Value input = adaptor.getInput();
    double p = adaptor.getP();
    int64_t axis = adaptor.getAxis();

    MultiDialectBuilder<MathBuilder, KrnlBuilder, MemRefBuilder,
        IndexExprBuilderForKrnl>
        create(rewriter, loc);
    IndexExprScope scope(create.krnl);

    // Prepare shape and type.
    MemRefType inputMemRefType = cast<MemRefType>(input.getType());
    Type elementType = inputMemRefType.getElementType();
    int64_t rank = inputMemRefType.getRank();

    if (axis < 0)
      axis += rank;

    // Compute dimensions.
    SmallVector<IndexExpr> dims;
    create.krnlIE.getShapeAsDims(input, dims);

    // Allocate output memref.
    MemRefType outputType = cast<MemRefType>(
        typeConverter->convertType(lpNormOp.getResult().getType()));
    Value output = create.mem.alignedAlloc(outputType, dims);

    // Allocate buffer for sum: same shape as input, but axis dim = 1
    SmallVector<IndexExpr> reducedDims(dims);
    reducedDims[axis] = LitIE(1);
    SmallVector<int64_t> reduceShape;
    for (auto d : reducedDims)
      reduceShape.push_back(
          d.isLiteral() ? d.getLiteral() : ShapedType::kDynamic);
    MemRefType reduceType = MemRefType::get(reduceShape, elementType);
    Value reduceMemRef = create.mem.alignedAlloc(reduceType, reducedDims);

    Value zero = create.math.constant(elementType, 0.0);
    create.krnl.memset(reduceMemRef, zero);

    // Compute norm value into reduceMemRef
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr> lbs(rank, LitIE(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, dims,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
          Value x = create.krnl.load(input, loopInd);
          Value val = nullptr;

          if (p == 1)
            val = create.math.abs(x);
          else if (p == 2)
            val = create.math.mul(x, x);
          else
            llvm_unreachable("Unsupported LpNorm order (only p=1 or p=2)");

          // Broadcast index
          SmallVector<Value> reduceIndex(loopInd.begin(), loopInd.end());
          reduceIndex[axis] = create.math.constantIndex(0);

          Value acc = create.krnl.load(reduceMemRef, reduceIndex);
          Value sum = create.math.add(acc, val);
          create.krnl.store(sum, reduceMemRef, reduceIndex);
        });

    // Normalize output = input / norm
    create.krnl.iterateIE(loopDef, loopDef, lbs, dims,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
          Value x = create.krnl.load(input, loopInd);

          SmallVector<Value> reduceIndex(loopInd.begin(), loopInd.end());
          reduceIndex[axis] = create.math.constantIndex(0);
          Value norm = create.krnl.load(reduceMemRef, reduceIndex);

          if (p == 2)
            norm = create.math.sqrt(norm);

          Value y = create.math.div(x, norm);
          create.krnl.store(y, output, loopInd);
        });

    rewriter.replaceOp(op, output);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXLpNormalizationOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLpNormalizationOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir