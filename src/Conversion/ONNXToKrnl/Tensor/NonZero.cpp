/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- NonZero.cpp - Lowering NonZero Op ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX NonZero Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXNonZeroOpLowering : public ConversionPattern {
  ONNXNonZeroOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXNonZeroOp::getOperationName(), 1, ctx) {}

  /// Given an input of shape (3, 2):
  /// [[2, 1],
  /// [0, 2],
  /// [0, 1]]
  ///
  /// Output will be: [[0, 0, 1, 1], [0, 1, 1, 1]]
  /// The output's shape is (2, 4) where 2 is the input's rank, 4 is the number
  /// of nonzero values in the input.
  ///
  /// Step 1: Compute a 0-1 matrix:
  /// [[1, 1],
  /// [0, 1],
  /// [0, 1]]
  ///
  /// Step 2: Compute reduction sum for each dimension:
  /// rsum0 = ReduceSum(axis = 1) = [2, 1, 1]
  /// rsum1 = ReduceSum(axis = 0) = [1, 3]
  ///
  /// Step 3: Compute the number of nonzero for allocating the output buffer.
  ///
  /// Step 4: Compute output for each dimension, e.g. for dimension 0:
  /// ```
  ///   k = 0
  ///   for i range(len(rsum0)):
  ///     d = rsum0[i]
  ///     for j in range(d):
  ///       out[0][k+j] = i
  ///     k += d
  /// ```
  ///
  /// Note: in the following implementation:
  /// - Step1, Step2, and Step 3 are done with a single nested loop so the 0-1
  /// matrix is not generated explicitly.
  ///
  /// - Computation in Step 4 is optimized for trip count, but invalid when
  /// using 'affine.for'. 'affine.for' does not allow using 'affine.for'
  /// operands as bounds in another 'affine.for'. More info:
  /// llvm-project/mlir/test/Dialect/Affine/invalid.mlir
  ///
  /// Thus, We rewrite the loop in Step 4 into an affine-compatible one as
  /// follows:
  /// ```
  /// for i in range(nonzerocount):
  ///   p = -1, s = 0
  ///   for j in range(len(rsum0)):
  ///      s += rsum0[j]
  ///      p = (i < s and p == -1) ? j : p
  ///   out[0][i] = p
  /// ```

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXNonZeroOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    // Builder helper.
    IndexExprScope outerScope(&rewriter, loc);
    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);

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
    MemRefBoundsIndexCapture xBounds(X);
    SmallVector<IndexExpr, 4> xLbs(xRank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> xUbs;
    xBounds.getDimList(xUbs);

    // Emit a variable for the total number of nonzero values.
    Value nonzeroCount = create.mem.alloca(MemRefType::get({}, indexTy));
    create.krnl.store(iZero, nonzeroCount, {});

    // Emit alloc and dealloc for reduction sum along each dimension.
    // MemRefType: [Dxi64] where D is the dimension size.
    SmallVector<Value, 4> rsumMemRefs;
    for (int i = 0; i < xRank; ++i) {
      // Alloc and dealloc.
      SmallVector<IndexExpr, 1> dimIE(1, xBounds.getDim(i));
      int64_t dim = dimIE[0].isLiteral() ? dimIE[0].getLiteral() : -1;
      Value alloc = insertAllocAndDeallocSimple(rewriter, op,
          MemRefType::get({dim}, indexTy), loc, dimIE,
          /*insertDealloc=*/true);
      // Initialize to zero.
      ValueRange initLoopDef = create.krnl.defineLoops(1);
      create.krnl.iterate(initLoopDef, initLoopDef, {iZero},
          {xBounds.getDim(i).getValue()},
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
        [&](KrnlBuilder &createKrnl, ValueRange iLoopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
              createKrnl);
          Value i(iLoopInd[0]);
          for (int64_t axis = 0; axis < xRank; ++axis) {
            Value axisVal = create.math.constantIndex(axis);
            MemRefBoundsIndexCapture rsumBounds(rsumMemRefs[axis]);

            create.krnl.store(iMinusOne, pos, {});
            create.krnl.store(iZero, sum, {});

            ValueRange jLoopDef = create.krnl.defineLoops(1);
            create.krnl.iterate(jLoopDef, jLoopDef, {iZero},
                {rsumBounds.getDim(0).getValue()},
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

    return success();
  }
};

void populateLoweringONNXNonZeroOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXNonZeroOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
