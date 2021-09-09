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
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXSplitV11OpLowering : public ConversionPattern {
  ONNXSplitV11OpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSplitV11Op::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    auto loc = op->getLoc();
    ONNXSplitV11OpAdaptor operandAdaptor(operands);
    ONNXSplitV11Op splitOp = llvm::dyn_cast<ONNXSplitV11Op>(op);
    auto rank = splitOp.input().getType().cast<ShapedType>().getRank();
    auto outputNum = splitOp.getNumResults();
    auto axis = splitOp.axis();

    // Get a shape helper.
    ONNXSplitV11OpShapeHelper shapeHelper(&splitOp, rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));

    // Alloc and dealloc.
    SmallVector<Value, 4> allocs;
    for (unsigned int i = 0; i < outputNum; ++i) {
      // Warning: insertDealloc is not used.
      bool insertDealloc = checkInsertDealloc(op, i);
      auto memRefType = convertToMemRefType(splitOp.outputs()[i].getType());
      Value alloc = insertAllocAndDeallocSimple(
          rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(i));
      allocs.emplace_back(alloc);
    }

    // Creates loops, one for each output.
    for (unsigned int i = 0; i < outputNum; ++i) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Create loop.
      BuildKrnlLoop outputLoops(rewriter, loc, rank);
      outputLoops.createDefineAndIterateOp(allocs[i]);
      rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

      // Scope for krnl ops
      IndexExprScope childScope(rewriter, shapeHelper.scope);
      KrnlBuilder createKrnl(rewriter, loc);

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
          for (unsigned int k = 0; k < i; ++k) {
            IndexExpr splitDim =
                SymbolIndexExpr(shapeHelper.dimsForOutput(k)[r]);
            readIndex = readIndex + splitDim;
          }
        }
        readIndices.emplace_back(readIndex);
        writeIndices.emplace_back(writeIndex);
      }
      // Insert copy.
      Value loadData = createKrnl.loadIE(operandAdaptor.input(), readIndices);
      createKrnl.storeIE(loadData, allocs[i], writeIndices);
    }
    rewriter.replaceOp(op, allocs);
    return success();
  }
};

void populateLoweringONNXSplitV11OpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSplitV11OpLowering>(ctx);
}
