/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- NonZero.cpp - Lowering NonZero Op ----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX NonZero Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXNonZeroOpLowering : public ConversionPattern {
  ONNXNonZeroOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXNonZeroOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXNonZeroOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    // Builder helper.
    KrnlBuilder createKrnl(rewriter, loc);
    IndexExprScope outerScope(rewriter, loc);

    // Common information.
    Value X = operandAdaptor.X();
    MemRefType xMemRefType = X.getType().cast<MemRefType>();
    MemRefType resMemRefType = convertToMemRefType(*op->result_type_begin());
    Type xElementType = xMemRefType.getElementType();
    Type resElementType = resMemRefType.getElementType();
    int64_t xRank = xMemRefType.getRank();

    // Constant values.
    Value iZero = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
    Value iOne = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);
    Value zero = emitConstantOp(rewriter, loc, xElementType, 0);
    LiteralIndexExpr litZero = LiteralIndexExpr(0);
    LiteralIndexExpr litRank = LiteralIndexExpr(xRank);

    // Bounds for iterating the input X.
    MemRefBoundsIndexCapture xBounds(X);
    SmallVector<IndexExpr, 4> lbs, ubs;
    for (decltype(xRank) i = 0; i < xRank; ++i) {
      lbs.emplace_back(litZero);
      ubs.emplace_back(xBounds.getDim(i));
    }

    // Compute the number of nonzero values.
    MemRefType i64MemRefType = MemRefType::get({}, rewriter.getIndexType());
    Value nonzeroCount = insertAllocAndDealloc(
        i64MemRefType, loc, rewriter, /*insertDealloc=*/true);
    createKrnl.store(iZero, nonzeroCount, {});

    // The result's first dimension size is the input's rank, so it is known.
    // The result's second dimension size is the number of nonzero values in the
    // input, and it is unknown at compile time. Thus, we will emit a
    // krnl.iterate to count the number of nonzero values.
    ValueRange countLoopDef = createKrnl.defineLoops(xMemRefType.getRank());
    createKrnl.iterateIE(countLoopDef, countLoopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          MathBuilder createMath(createKrnl);
          Value val = createKrnl.load(X, loopInd);
          Value eqCond = createMath.eq(val, zero);
          Value zeroOrOne = createMath.select(eqCond, iZero, iOne);
          Value counter = createKrnl.load(nonzeroCount, {});
          Value newCount = createMath.add(counter, zeroOrOne);
          createKrnl.store(newCount, nonzeroCount);
        });

    // Insert an allocation and deallocation for the result of this operation.
    Value numberOfZeros = createKrnl.load(nonzeroCount, {});
    SmallVector<IndexExpr, 2> dimExprs;
    dimExprs.emplace_back(litRank);
    dimExprs.emplace_back(DimIndexExpr(numberOfZeros));
    bool insertDealloc = checkInsertDealloc(op);
    Value resMemRef = insertAllocAndDeallocSimple(
        rewriter, op, resMemRefType, loc, dimExprs, insertDealloc);

    // When computing indices of nonzero value, keep trace of current indices
    // for each dimension so that we know exactly where to store in the result.
    Value pos =
        insertAllocAndDealloc(MemRefType::get({xRank}, rewriter.getIndexType()),
            loc, rewriter, /*insertDealloc=*/true);
    ValueRange posLoopDef = createKrnl.defineLoops(1);
    createKrnl.iterateIE(posLoopDef, posLoopDef, {litZero}, {litRank},
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          createKrnl.store(iZero, pos, loopInd);
        });

    // Iterate over the input in row-major order to get indices of nonzero
    // values.
    ValueRange mainLoopDef = createKrnl.defineLoops(xMemRefType.getRank());
    createKrnl.iterateIE(mainLoopDef, mainLoopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          MathBuilder createMath(createKrnl);
          Value val = createKrnl.load(X, loopInd);
          Value eqCond = createMath.eq(val, zero);
          Value zeroOrOne = createMath.select(eqCond, iZero, iOne);
          for (decltype(xRank) i = 0; i < xRank; ++i) {
            SmallVector<IndexExpr, 1> posInd;
            posInd.emplace_back(LiteralIndexExpr(i));
            // Load the current position along dimension i.
            Value currentPos = createKrnl.loadIE(pos, posInd);
            // Load the output at the current position along dimension i.
            SmallVector<IndexExpr, 2> resInd;
            resInd.emplace_back(LiteralIndexExpr(i));
            resInd.emplace_back(DimIndexExpr(currentPos));
            // If value is nonzero, update the output with the value's index.
            // Otherwise, keep the output unchanged.
            Value oldIndex = createKrnl.loadIE(resMemRef, resInd);
            Value newIndex =
                rewriter.create<IndexCastOp>(loc, loopInd[i], resElementType);
            newIndex = createMath.select(eqCond, oldIndex, newIndex);
            createKrnl.storeIE(newIndex, resMemRef, resInd);
            // Move forward along the current dimension.
            currentPos = createMath.add(currentPos, zeroOrOne);
            createKrnl.storeIE(currentPos, pos, posInd);
          }
        });

    rewriter.replaceOp(op, resMemRef);

    return success();
  }
};

void populateLoweringONNXNonZeroOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXNonZeroOpLowering>(ctx);
}
