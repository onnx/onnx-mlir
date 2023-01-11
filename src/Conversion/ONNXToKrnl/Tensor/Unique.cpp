/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Unique.cpp - Lowering Unique Op ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Unique Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXUniqueOpLowering : public ConversionPattern {
  ONNXUniqueOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXUniqueOp::getOperationName(), 1, ctx) {}

  ///
  /// XXX Explanation about operation and steps is required here.
  ///

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
#if 0
    ONNXUniqueOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    // Builder helper.
    IndexExprScope outerScope(&rewriter, loc);
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Frequently used MemRefType.
    Value X = operandAdaptor.X();
    MemRefType xMemRefType = X.getType().cast<MemRefType>();
    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType resMemRefType = convertedType.cast<MemRefType>();
    int64_t xRank = xMemRefType.getRank();

    // Frequently used element types.
    Type indexTy = rewriter.getIndexType();
    Type xElementType = xMemRefType.getElementType();
    Type resElementType = resMemRefType.getElementType();

    // Constant values.
    Value iZero = create.math.constantIndex(0);
    Value iOne = create.math.constantIndex(1);
    Value iMinusOne = create.math.constantIndex(-1);
    Value zero = create.math.constant(xElementType, 0);

    // Bounds for the input tensor.
    SmallVector<IndexExpr, 4> xLbs(xRank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> xUbs;
    create.krnlIE.getShapeAsDims(X, xUbs);

    // Emit a variable for the total number of nonzero values.
    Value nonzeroCount = create.mem.alloca(MemRefType::get({}, indexTy));
    create.krnl.store(iZero, nonzeroCount, {});

    // Emit alloc and dealloc for reduction sum along each dimension.
    // MemRefType: [Dxi64] where D is the dimension size.
    SmallVector<Value, 4> rsumMemRefs;
    for (int i = 0; i < xRank; ++i) {
      // Alloc and dealloc.
      IndexExpr xBound = create.krnlIE.getShapeAsDim(X, i);
      SmallVector<IndexExpr, 1> dimIE(1, xBound);
      int64_t dim = dimIE[0].isLiteral() ? dimIE[0].getLiteral() : -1;
      Value alloc = insertAllocAndDeallocSimple(rewriter, op,
          MemRefType::get({dim}, indexTy), loc, dimIE,
          /*insertDealloc=*/true);
      // Initialize to zero.
      ValueRange initLoopDef = create.krnl.defineLoops(1);
      create.krnl.iterate(initLoopDef, initLoopDef, {iZero},
          {xBound.getValue()},
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            createKrnl.store(iZero, alloc, loopInd);
          });
      rsumMemRefs.emplace_back(alloc);
    }

    // Emit a loop for counting the total number of nonzero values, and
    // the reduction sum for each dimension.
    ValueRange rsumLoopDef = create.krnl.defineLoops(xMemRefType.getRank());
    create.krnl.iterateIE(rsumLoopDef, rsumLoopDef, xLbs, xUbs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          MathBuilder createMath(createKrnl);
          Value x = createKrnl.load(X, loopInd);
          Value eqCond = createMath.eq(x, zero);
          Value zeroOrOne = createMath.select(eqCond, iZero, iOne);
          // Count the total number of nonzero values.
          Value total = createKrnl.load(nonzeroCount, {});
          total = createMath.add(total, zeroOrOne);
          createKrnl.store(total, nonzeroCount, {});
          // Reduction sum of the number of nonzero values for each dimension.
          for (int64_t i = 0; i < xRank; ++i) {
            Value sum = createKrnl.load(rsumMemRefs[i], loopInd[i]);
            sum = createMath.add(sum, zeroOrOne);
            createKrnl.store(sum, rsumMemRefs[i], loopInd[i]);
          }
        });

    // Emit alloc and dealloc for the result of this operation.
    // MemRefType : [RxNxi64] where R is the input's rank, N is the number of
    // non zero values.
    Value numberOfZeros = create.krnl.load(nonzeroCount, {});
    SmallVector<IndexExpr, 2> dimExprs;
    dimExprs.emplace_back(LiteralIndexExpr(xRank));
    dimExprs.emplace_back(DimIndexExpr(numberOfZeros));
    bool insertDealloc = checkInsertDealloc(op);
    Value resMemRef = insertAllocAndDeallocSimple(
        rewriter, op, resMemRefType, loc, dimExprs, insertDealloc);

    // Emit code to compute the output for each dimension.
    // ```
    // for i in range(nonzerocount):
    //   p = -1, s = 0
    //   for j in range(len(rsum0)):
    //      s += rsum0[j]
    //      p = (i < s and p == -1) ? j : p
    //   out[0][i] = p
    // ```

    Value pos = create.mem.alloca(MemRefType::get({}, indexTy));
    Value sum = create.mem.alloca(MemRefType::get({}, indexTy));
    ValueRange iLoopDef = create.krnl.defineLoops(1);
    create.krnl.iterate(iLoopDef, iLoopDef, {iZero}, {numberOfZeros},
        [&](KrnlBuilder &ck, ValueRange iLoopInd) {
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
              MemRefBuilder>
              create(ck);
          Value i(iLoopInd[0]);
          for (int64_t axis = 0; axis < xRank; ++axis) {
            Value axisVal = create.math.constantIndex(axis);
            Value rsumBoundsVal = rsumMemRefs[axis];
            IndexExpr rsumBounds0 =
                create.krnlIE.getShapeAsDim(rsumBoundsVal, 0);

            create.krnl.store(iMinusOne, pos, {});
            create.krnl.store(iZero, sum, {});

            ValueRange jLoopDef = create.krnl.defineLoops(1);
            create.krnl.iterate(jLoopDef, jLoopDef, {iZero},
                {rsumBounds0.getValue()},
                [&](KrnlBuilder &createKrnl, ValueRange jLoopInd) {
                  MathBuilder createMath(createKrnl);
                  Value j(jLoopInd[0]);
                  Value o = createKrnl.load(rsumMemRefs[axis], {j});
                  Value s = createKrnl.load(sum, {});
                  Value p = createKrnl.load(pos, {});
                  s = createMath.add(s, o);
                  Value andCond = createMath.andi(
                      createMath.slt(i, s), createMath.eq(p, iMinusOne));
                  p = createMath.select(andCond, j, p);
                  createKrnl.store(p, pos, {});
                  createKrnl.store(s, sum, {});
                });
            Value p = create.krnl.load(pos, {});
            p = create.math.cast(resElementType, p);
            create.krnl.store(p, resMemRef, {axisVal, i});
          }
        });

    rewriter.replaceOp(op, resMemRef);
#endif

#if 0
    Location loc = op->getLoc();
    ONNXTopKOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Value X = operandAdaptor.X();

    // Builders.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
        rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType resMemRefType = convertedType.cast<MemRefType>();

    // Common types.
    Type i64Type = rewriter.getI64Type();

    // Op's Attributes.
    int64_t rank = resMemRefType.getRank();
    int64_t axis = operandAdaptor.axis();
    axis = axis < 0 ? axis + rank : axis;
    assert(axis >= 0 && axis < rank && "axis is out of bound");
    bool ascendingMode = operandAdaptor.largest() != 1;
    // According to ONNX TopK: 'If "sorted" is 0, order of returned 'Values' and
    // 'Indices' are undefined'.
    // In this case, we still return sorted values and indices to make them
    // deterministic. So this attribute is not used.
    // bool sortedMode = TopKOp.sorted() == 1;

    // Compute the output's dimension sizes.
    ONNXTopKOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr resDims = shapeHelper.getOutputDims();

    // Insert an allocation and deallocation for the results of this operation.
    bool insertDealloc = checkInsertDealloc(op, /*resultIndex=*/0);
    Value resMemRef = insertAllocAndDeallocSimple(
        rewriter, op, resMemRefType, loc, resDims, insertDealloc);
    insertDealloc = checkInsertDealloc(op, /*resultIndex=*/1);
    Value resIndexMemRef = insertAllocAndDeallocSimple(rewriter, op,
        MemRefType::get(resMemRefType.getShape(), i64Type), loc, resDims,
        insertDealloc);

    // Compute argSort of X along axis.
    Value argSort =
        emitArgSort(rewriter, loc, X, axis, /*ascending=*/ascendingMode);


    // Produce the final result.
    SmallVector<IndexExpr> zeroDims(rank, LiteralIndexExpr(0));
    ValueRange loopDef = create.krnl.defineLoops(rank);
    create.krnl.iterateIE(loopDef, loopDef, zeroDims, resDims,
        [&](KrnlBuilder &createKrnl, ValueRange resLoopInd) {
          Value resInd = createKrnl.load(argSort, resLoopInd);
          SmallVector<Value> resIndexLoopInd(resLoopInd);
          resIndexLoopInd[axis] = resInd;
          // Store value.
          Value val = createKrnl.load(X, resIndexLoopInd);
          createKrnl.store(val, resMemRef, resLoopInd);
          // Store index.
          Value resIndI64 =
              rewriter.create<arith::IndexCastOp>(loc, i64Type, resInd);
          createKrnl.store(resIndI64, resIndexMemRef, resLoopInd);
        });

    rewriter.replaceOp(op, {resMemRef, resIndexMemRef});
#endif

    return success();
  }
};

void populateLoweringONNXUniqueOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXUniqueOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
