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

template <typename Adaptor, typename Op, typename ShapeHelper>
LogicalResult ONNXSplitOpLoweringCommon(Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) {
  // Gather info.
  auto loc = op->getLoc();
  Adaptor operandAdaptor(operands);
  Op splitOp = llvm::dyn_cast<Op>(op);
  auto rank = splitOp.input().getType().template cast<ShapedType>().getRank();
  auto outputNum = splitOp.getNumResults();
  auto axis = splitOp.axis();

  // Get a shape helper.
  ShapeHelper shapeHelper(&splitOp, rewriter,
      getDenseElementAttributeFromKrnlValue, loadDenseElementArrayValueAtIndex);
  auto shapecomputed = shapeHelper.Compute(operandAdaptor);
  assert(succeeded(shapecomputed));

  // Alloc and dealloc.
  SmallVector<Value, 4> allocs;
  for (unsigned int i = 0; i < outputNum; ++i) {
    checkInsertDealloc(op, i);
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
          IndexExpr splitDim = SymbolIndexExpr(shapeHelper.dimsForOutput(k)[r]);
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

struct ONNXSplitOpLowering : public ConversionPattern {
  ONNXSplitOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSplitOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSplitOpLoweringCommon<ONNXSplitOpAdaptor, ONNXSplitOp,
        ONNXSplitOpShapeHelper>(op, operands, rewriter);
  }
};

struct ONNXSplitV11OpLowering : public ConversionPattern {
  ONNXSplitV11OpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSplitV11Op::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSplitOpLoweringCommon<ONNXSplitV11OpAdaptor, ONNXSplitV11Op,
        ONNXSplitV11OpShapeHelper>(op, operands, rewriter);
  }
};

void populateLoweringONNXSplitOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLowering>(ctx);
}

void populateLoweringONNXSplitV11OpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSplitV11OpLowering>(ctx);
}
