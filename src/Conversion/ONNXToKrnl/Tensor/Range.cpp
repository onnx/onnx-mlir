/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Range.cpp - Lowering Range Op --------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Range Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXRangeOpLowering : public ConversionPattern {
  ONNXRangeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXRangeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXRangeOpAdaptor operandAdaptor(operands);
    ONNXRangeOp rangeOp = dyn_cast_or_null<ONNXRangeOp>(op);
    auto loc = op->getLoc();

    // Create an index expression scope.
    // Scope for krnl EDSC ops
    using namespace mlir::edsc;
    ScopedContext scope(rewriter, loc);
    // Scope for IndexExpr.
    IndexExprScope ieScope(&rewriter, loc);

    Value start = operandAdaptor.start();
    Value limit = operandAdaptor.limit();
    Value delta = operandAdaptor.delta();

    auto startShape = start.getType().cast<MemRefType>().getShape();
    auto limitShape = limit.getType().cast<MemRefType>().getShape();
    auto deltaShape = delta.getType().cast<MemRefType>().getShape();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto memRefShape = memRefType.getShape();

    // Allocate result.
    Value alloc;
    Value loadedStart;
    Value loadedDelta;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);

      // TODO:
      // loadedStart
      // loadedDelta
    } else {
      Value zero = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
      Value loadedLimit = rewriter.create<KrnlLoadOp>(loc, limit, zero);
      loadedStart = rewriter.create<KrnlLoadOp>(loc, start, zero);
      loadedDelta = rewriter.create<KrnlLoadOp>(loc, delta, zero);

      Value elements = rewriter.create<DivFOp>(loc,
          rewriter.create<SubFOp>(loc, loadedLimit, loadedStart), loadedDelta);

      Value numberOfElements = rewriter.create<IndexCastOp>(loc,
          rewriter.create<mlir::FPToUIOp>(loc,
              rewriter.create<mlir::CeilFOp>(loc, elements),
              rewriter.getIntegerType(64)),
          rewriter.getIndexType());
      SmallVector<Value, 4> allocOperands;
      allocOperands.push_back(numberOfElements);
      memref::AllocOp allocateMemref =
          rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      alloc = allocateMemref;
    }

    // Create a single loop.
    BuildKrnlLoop krnlLoop(rewriter, loc, 1);

    // Emit the definition.
    krnlLoop.createDefineOp();

    SmallVector<int64_t, 1> accShape;
    accShape.emplace_back(1);
    auto accType = MemRefType::get(accShape, rewriter.getF32Type());
    auto acc = rewriter.create<memref::AllocOp>(loc, accType);

    // Acc index:
    SmallVector<IndexExpr, 4> accIndex;
    accIndex.emplace_back(LiteralIndexExpr(0));

    // Initialize accumulator with value:
    krnl_store(loadedStart, acc, accIndex);

    // Emit body of the loop:
    // output[i] = start + (i * delta);
    int nIndex = krnlLoop.pushBounds(0, alloc, 0);
    krnlLoop.createIterateOp();
    rewriter.setInsertionPointToStart(krnlLoop.getIterateBlock());
    {
      // Read value:
      Value result = krnl_load(acc, accIndex);

      // Store result:
      SmallVector<IndexExpr, 4> resultIndices;
      resultIndices.emplace_back(
          DimIndexExpr(krnlLoop.getInductionVar(nIndex)));
      krnl_store(result, alloc, resultIndices);

      // Increment result:
      Value accResult = rewriter.create<AddFOp>(loc, result, loadedDelta);
      krnl_store(accResult, acc, accIndex);
    }

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXRangeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRangeOpLowering>(ctx);
}
