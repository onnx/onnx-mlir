/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- Trilu.cpp - TriluOp ---------------------------===//
//
// Copyright 2023- The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX trilu operator to Krnl dialect.
//
// It looks like this operator can be lowered via the lowering pattern for unary
// operators if the lowering pattern for unary operators is extended to take
// loop indices as inputs in their scalar computation.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXTriluOpLowering : public OpConversionPattern<ONNXTriluOp> {
  ONNXTriluOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}
  LogicalResult matchAndRewrite(ONNXTriluOp triluOp, ONNXTriluOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = triluOp.getOperation();
    Location loc = ONNXLoc<ONNXTriluOp>(op);
    Value input = adaptor.getInput();
    bool retainUpper = adaptor.getUpper() == 1;

    MultiDialectBuilder<MathBuilder, KrnlBuilder, IndexExprBuilderForKrnl,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnl);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

    int64_t rank = memRefType.getRank();
    Type elementType = memRefType.getElementType();
    Value zero = create.math.constant(elementType, 0.0);

    // Load k value.
    Value k;
    if (isNoneValue(triluOp.getK()))
      k = create.math.constantIndex(0);
    else
      k = create.math.castToIndex(create.krnl.load(adaptor.getK()));

    // Insert an allocation and deallocation for the result of this operation.
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(input, ubs);
    Value resMemRef = create.mem.alignedAlloc(memRefType, ubs);

    // Main loop.
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr> lbs(rank, LitIE(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder> create(
              createKrnl);
          Value i = create.math.add(k, loopInd[rank - 2]);
          Value j = loopInd[rank - 1];
          Value setToZero =
              retainUpper ? create.math.gt(i, j) : create.math.lt(i, j);

          Value loadVal = create.krnl.load(input, loopInd);
          Value storeVal = create.math.select(setToZero, zero, loadVal);
          create.krnl.store(storeVal, resMemRef, loopInd);
        });

    rewriter.replaceOp(op, resMemRef);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXTriluOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTriluOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
