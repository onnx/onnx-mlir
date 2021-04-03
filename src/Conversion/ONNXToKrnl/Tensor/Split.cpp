/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Split.cpp - Lowering Split Op -----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Split Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlIntrinsics.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXSplitOpLowering : public ConversionPattern {
  ONNXSplitOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSplitOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    auto loc = op->getLoc();
    ONNXSplitOpAdaptor operandAdaptor(operands);
    ONNXSplitOp splitOp = llvm::dyn_cast<ONNXSplitOp>(op);
    auto rank = splitOp.input().getType().cast<ShapedType>().getRank();
    auto outputNum = splitOp.getNumResults();
    auto axis = splitOp.axis();

    // Get a shape helper.
    ONNXSplitOpShapeHelper shapeHelper(&splitOp, &rewriter);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));

    // Alloc and dealloc.
    SmallVector<Value, 4> allocs;
    for (int i = 0; i < outputNum; ++i) {
      bool insertDealloc = checkInsertDealloc(op, i);
      auto memRefType = convertToMemRefType(splitOp.outputs()[i].getType());
      Value alloc = insertAllocAndDeallocSimple(
          rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(i));
      allocs.emplace_back(alloc);
    }

    // Creates loops, one for each output.
    for (int i = 0; i < outputNum; ++i) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Create loop.
      BuildKrnlLoop outputLoops(rewriter, loc, rank);
      outputLoops.createDefineAndIterateOp(allocs[i]);
      rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

      // Scope for krnl EDSC ops
      using namespace edsc;
      // using namespace edsc::intrinsics;
      ScopedContext scope(rewriter, loc);
      IndexExprScope childScope(shapeHelper.scope);

      // Indices for the read and write.
      SmallVector<IndexExpr, 4> readIndices;
      SmallVector<IndexExpr, 4> writeIndices;
      for (int r = 0; r < rank; ++r) {
        Value readVal = outputLoops.getInductionVar(r);
        // If not the split axis, same index for read and write
        IndexExpr readIndex = DimIndexExpr(readVal);
        DimIndexExpr writeIndex(readVal);
        // If the split axis, compute read index for the split axis.
        if (r == axis) {
          for (int k = 0; k < i; ++k) {
            IndexExpr splitDim =
                SymbolIndexExpr(shapeHelper.dimsForOutput(k)[r]);
            readIndex = readIndex + splitDim;
          }
        }
        readIndices.emplace_back(readIndex);
        writeIndices.emplace_back(writeIndex);
      }
      // Insert copy.
      Value loadData = krnl_load(operandAdaptor.input(), readIndices);
      krnl_store(loadData, allocs[i], writeIndices);
    }
    rewriter.replaceOp(op, allocs);
    return success();
  }
};

void populateLoweringONNXSplitOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLowering>(ctx);
}
