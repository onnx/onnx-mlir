/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Concat Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXConcatOpLowering : public ConversionPattern {
  ONNXConcatOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConcatOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    auto loc = op->getLoc();

    ONNXConcatOpAdaptor operandAdaptor(operands);
    ONNXConcatOp concatOp = llvm::cast<ONNXConcatOp>(op);
    ONNXConcatOpShapeHelper shapeHelper(&concatOp, &rewriter);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    (void)shapecomputed;
    assert(succeeded(shapecomputed));

    auto axis = concatOp.axis();
    int inputNum = operands.size();

    // Alloc and dealloc.
    auto resultOperand = concatOp.concat_result();
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto resultShape = outputMemRefType.getShape();
    auto rank = resultShape.size();

    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));
    ;

    // Creates loops, one for each input.
    for (int i = 0; i < inputNum; ++i) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Operand info.
      auto currShape = operands[i].getType().cast<MemRefType>().getShape();
      // Create loop.
      BuildKrnlLoop inputLoops(rewriter, loc, rank);
      inputLoops.createDefineOp();
      for (int r = 0; r < rank; ++r)
        inputLoops.pushBounds(0, operands[i], r);
      inputLoops.createIterateOp();
      rewriter.setInsertionPointToStart(inputLoops.getIterateBlock());

      // Indices for the read and write.
      SmallVector<Value, 4> readIndices;
      SmallVector<Value, 4> writeIndices;
      for (int r = 0; r < rank; ++r) {
        readIndices.emplace_back(inputLoops.getInductionVar(r));
        if (r != axis || i == 0) {
          writeIndices.emplace_back(inputLoops.getInductionVar(r));
        } else {
          IndexExprScope IEScope(&rewriter, loc);
          IndexExpr writeOffset = DimIndexExpr(inputLoops.getInductionVar(r));
          for (int j = 0; j < i; j++) {
            MemRefBoundIndexCapture operandJBounds(operands[j]);
            writeOffset = writeOffset + operandJBounds.getDim(r);
          }
          writeIndices.emplace_back(writeOffset.getValue());
        }
      }
      // Insert copy.
      auto loadData =
          rewriter.create<KrnlLoadOp>(loc, operands[i], readIndices);
      rewriter.create<KrnlStoreOp>(loc, loadData, alloc, writeIndices);
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXConcatOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConcatOpLowering>(ctx);
}
