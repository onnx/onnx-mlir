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
  /// 1, 1
  /// 0, 1
  /// 0, 1
  ///
  /// Step 2: Compute reduction sum for each dimension:
  /// dim0 = ReduceSum(axis = 1) = [2, 1, 1]
  /// dim1 = ReduceSum(axis = 0) = [1, 3]
  ///
  /// Step 3: Compute the number of nonzero for allocating the output buffer.
  ///
  /// Step 4: Compute output for each dimension:
  /// for each axis:
  ///   k = 0
  ///   for i range(len(dim0)):
  ///     d = dim0[i]
  ///     for j in range(d):
  ///       out[0][k+j] = i
  ///     k += d
  ///
  /// Note: in the following implementation:
  /// - Step1 and Step2 are done with a single nested loop for each dimension,
  ///   so the 0-1 matrix is not generated explicitly.
  /// - Step 3 will be done in one of the reduction sums in Step 2. Try to
  ///   select the reduction sum with the smallest dim size if possible.

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXNonZeroOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    // Builder helper.
    IndexExprScope outerScope(rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);
    MemRefBuilder createMemRef(createKrnl);

    // Frequently used MemRefType.
    Value X = operandAdaptor.X();
    MemRefType xMemRefType = X.getType().cast<MemRefType>();
    MemRefType resMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t xRank = xMemRefType.getRank();

    // Frequently used element types.
    Type indexTy = rewriter.getIndexType();
    Type xElementType = xMemRefType.getElementType();
    Type resElementType = resMemRefType.getElementType();

    // Constant values.
    Value iZero = emitConstantOp(rewriter, loc, indexTy, 0);
    Value iOne = emitConstantOp(rewriter, loc, indexTy, 1);
    Value zero = emitConstantOp(rewriter, loc, xElementType, 0);
    LiteralIndexExpr litZero(0);

    // Bounds for iterating the input X.
    MemRefBoundsIndexCapture xBounds(X);
    SmallVector<IndexExpr, 4> xLbs, xUbs;
    for (int64_t i = 0; i < xRank; ++i) {
      xLbs.emplace_back(litZero);
      xUbs.emplace_back(xBounds.getDim(i));
    }

    // Find the axis with the smallest dimension size in the input.
    // Reduction sum for this axis will be used to compute the number of nonzero
    int64_t smallestDimAxis = 0;
    if (xMemRefType.hasStaticShape() && xMemRefType.getRank() != 0) {
      ArrayRef<int64_t> shape = xMemRefType.getShape();
      int64_t v = shape[0];
      for (uint64_t i = 1; i < shape.size(); i++) {
        if (shape[i] < v) {
          v = shape[i];
          smallestDimAxis = i;
        }
      }
    }

    // Emit alloc and dealloc for reduction sum along each axis
    // MemRefType: [Dxi64] where D is dimension size of an axis.
    SmallVector<Value, 4> redMemRefs;
    for (int i = 0; i < xRank; ++i) {
      // Alloc and dealloc.
      SmallVector<IndexExpr, 1> dimIE(1, xBounds.getDim(i));
      int64_t dim = dimIE[0].isLiteral() ? dimIE[0].getLiteral() : -1;
      Value alloc = insertAllocAndDeallocSimple(rewriter, op,
          MemRefType::get({dim}, indexTy), loc, dimIE,
          /*insertDealloc=*/true);
      // Initialize to zero.
      ValueRange initLoopDef = createKrnl.defineLoops(1);
      createKrnl.iterateIE(initLoopDef, initLoopDef, {litZero}, dimIE,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            createKrnl.store(iZero, alloc, loopInd);
          });
      redMemRefs.emplace_back(alloc);
    }

    // Emit a loop for reduction sums along each axis.
    ValueRange redLoopDef = createKrnl.defineLoops(xMemRefType.getRank());
    createKrnl.iterateIE(redLoopDef, redLoopDef, xLbs, xUbs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          MathBuilder createMath(createKrnl);
          Value val = createKrnl.load(X, loopInd);
          Value eqCond = createMath.eq(val, zero);
          Value zeroOrOne = createMath.select(eqCond, iZero, iOne);
          for (int64_t i = 0; i < xRank; ++i) {
            Value sum = createKrnl.load(redMemRefs[i], loopInd[i]);
            sum = createMath.add(sum, zeroOrOne);
            createKrnl.store(sum, redMemRefs[i], loopInd[i]);
          }
        });

    // Emit a variable for the number of nonzero values.
    Value nonzeroCount = createMemRef.alloca(MemRefType::get({}, indexTy));
    createKrnl.store(iZero, nonzeroCount, {});

    // Emit code to compute the number of nonzero values, using the reduction
    // sum of the smalles size.
    ValueRange countLoopDef = createKrnl.defineLoops(1);
    createKrnl.iterateIE(countLoopDef, countLoopDef, {litZero},
        xUbs[smallestDimAxis],
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          MathBuilder createMath(createKrnl);
          Value sum = createKrnl.load(
              redMemRefs[smallestDimAxis], loopInd[smallestDimAxis]);
          Value count = createKrnl.load(nonzeroCount, {});
          sum = createMath.add(sum, count);
          createKrnl.store(sum, nonzeroCount, {});
        });

    // Emit alloc and dealloc for the result of this operation.
    // MemRefType : [RxNxi64] where R is the input's rank, N is the number of
    // non zero values.
    Value numberOfZeros = createKrnl.load(nonzeroCount, {});
    SmallVector<IndexExpr, 2> dimExprs;
    dimExprs.emplace_back(LiteralIndexExpr(xRank));
    dimExprs.emplace_back(DimIndexExpr(numberOfZeros));
    bool insertDealloc = checkInsertDealloc(op);
    Value resMemRef = insertAllocAndDeallocSimple(
        rewriter, op, resMemRefType, loc, dimExprs, insertDealloc);

    // Emit code to compute the output for each axis.
    // for each axis:
    //   k = 0
    //   for i range(len(dim0)):
    //     d = dim0[i]
    //     for j in range(d):
    //       out[0][k+j] = i
    //     k += d
    Value kMemRef = createMemRef.alloca(MemRefType::get({}, indexTy));
    for (int64_t axis = 0; axis < xRank; ++axis) {
      Value outInd0 = emitConstantOp(rewriter, loc, indexTy, axis);
      createKrnl.store(iZero, kMemRef, {});
      // for i range(len(dim0)):
      ValueRange iLoopDef = createKrnl.defineLoops(1);
      createKrnl.iterateIE(iLoopDef, iLoopDef, {litZero}, {xUbs[axis]},
          [&](KrnlBuilder &createKrnl, ValueRange iLoopInd) {
            MathBuilder createMath(createKrnl);
            Value i(iLoopInd[0]);
            Value iVal = rewriter.create<IndexCastOp>(loc, i, resElementType);
            Value d = createKrnl.load(redMemRefs[axis], {i});
            Value k = createKrnl.load(kMemRef, {});
            // for j in range(d):
            ValueRange jLoopDef = createKrnl.defineLoops(1);
            createKrnl.iterate(jLoopDef, jLoopDef, {iZero}, {d},
                [&](KrnlBuilder &createKrnl, ValueRange jLoopInd) {
                  MathBuilder createMath(createKrnl);
                  Value j(jLoopInd[0]);
                  Value outInd1 = createMath.add(k, j);
                  // out[0][k+j] = i
                  createKrnl.store(iVal, resMemRef, {outInd0, outInd1});
                });
            // k += d
            k = createMath.add(k, d);
            createKrnl.store(k, kMemRef, {});
          });
    }

    rewriter.replaceOp(op, resMemRef);

    return success();
  }
};

void populateLoweringONNXNonZeroOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXNonZeroOpLowering>(ctx);
}
