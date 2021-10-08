/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Gather.cpp - Lowering Gather Op----------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXReverseSequenceOpLowering : public ConversionPattern {
  ONNXReverseSequenceOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXReverseSequenceOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXReverseSequenceOpAdaptor operandAdaptor(operands);
    ONNXReverseSequenceOp reverseSequenceOp =
        llvm::cast<ONNXReverseSequenceOp>(op);
    auto loc = op->getLoc();

    ONNXReverseSequenceOpShapeHelper shapeHelper(&reverseSequenceOp, rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));
    // Scope for krnl ops
    IndexExprScope outerScope(rewriter, shapeHelper.scope);
    KrnlBuilder createKrnl(rewriter, loc);

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Save axis and rank info.
    int64_t batchAxis = reverseSequenceOp.batch_axis();
    int64_t timeAxis = reverseSequenceOp.time_axis();

    MemRefBoundsIndexCapture dataBounds(operandAdaptor.input());
    int64_t outputRank = shapeHelper.dimsForOutput(0).size();

    /*
      The semantic of Reversequence can be expressed in loop as:
      for (vector of induction I on output dimension) {
        // Assume it is index of time_axis dimension
        Iinput = I;
        Iinput[time_axis()]  = I[time_axis] < seqence_lens[I[batch_axis]]?
              sequence_lens[I[batch_axis]]-I[time_axis]-1:I[time_axis()];
        output[I] = input[Iinput]
      }

      The loop can be transformed as:
      for (I[batch_axis()) {
        for (I[time_axis() from 0 to sequence_lens[I[batch_axis]) {
          for (other elment of I ) {
            Iinput = I;
            Iinput[time_axis()] = sequence_lens[I[batch_axis]]-I[time_axis]-1
            output[I] = input[Iinput];
          }
        for (I[time_axis() from sequence_lens[I[batch_axis] to end) {
          for (other elment of I ) {
              output[I] = input[I];
          }
        }
      }
    }
    */
    // Define loops and iteration trip counts (equivalent to size of output)
    BuildKrnlLoop outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    outputLoops.createIterateOp();

    // Insert code inside the loop.
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());
    IndexExprScope innerLoopScope(rewriter, outerScope);

    LiteralIndexExpr one(1);

    // LiteralIndexExpr batch(axisLit);
    // SymbolIndexExpr axisDim(shapeHelper.dataDims[axisLit]);

    // compute the loop indices for the output
    SmallVector<IndexExpr, 4> outputAccessFct;
    getIndexExprList<DimIndexExpr>(
        outputLoops.getAllInductionVar(), outputAccessFct);

    // Compute access function for indices[jjs].
    SmallVector<IndexExpr, 4> inputAccessFct;
    getIndexExprList<DimIndexExpr>(
        outputLoops.getAllInductionVar(), inputAccessFct);
    Value lensVal = createKrnl.loadIE(
        operandAdaptor.sequence_lens(), inputAccessFct[batchAxis]);
    IndexExpr lens = NonAffineIndexExpr(lensVal);
    DimIndexExpr timeDim = inputAccessFct[timeAxis];
    IndexExpr inputIndex =
        IndexExpr::select(timeDim < lens, lens - timeDim - one, timeDim);
    inputAccessFct[timeAxis] = inputIndex;
    Value inputVal = createKrnl.loadIE(operandAdaptor.input(), inputAccessFct);

    // Save data into output
    createKrnl.storeIE(inputVal, alloc, outputAccessFct);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXReverseSequenceOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReverseSequenceOpLowering>(ctx);
}
