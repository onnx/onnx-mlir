/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Hardmax.cpp - Hardmax Op ---------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to Krnl dialect.
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

    // TODO: support k != 0.
    MultiDialectBuilder<MathBuilder, KrnlBuilder, IndexExprBuilderForKrnl,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnl);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();

    int64_t rank = memRefType.getRank();
    Type elementType = memRefType.getElementType();
    Value zero = create.math.constant(elementType, 0.0);

    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(input, ubs);

    // Insert an allocation and deallocation for the result of this operation.
    Value resMemRef = create.mem.alignedAlloc(memRefType, ubs);

    // Main loop.
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr> lbs(rank, LiteralIndexExpr(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder> create(
              createKrnl);
          Value val = create.krnl.load(input, loopInd);
          // Set value to 1 if the index is argmax. Otherwise, 0.
          Value i = loopInd[rank - 2];
          Value j = loopInd[rank - 1];
          Value retainCond;
          if (retainUpper)
            retainCond = create.math.gt(i, j);
          else
            retainCond = create.math.lt(i, j);
          create.scf.ifThenElse(
              retainCond, /*then*/
              [&](SCFBuilder &createSCF) {
                MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createSCF);
                create.krnl.store(zero, resMemRef, loopInd);
              },
              /*else*/
              [&](SCFBuilder &createSCF) {
                MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createSCF);
                create.krnl.store(val, resMemRef, loopInd);
              });
        });

    rewriter.replaceOp(op, resMemRef);
    return success();
  }
};

void populateLoweringONNXTriluOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTriluOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
