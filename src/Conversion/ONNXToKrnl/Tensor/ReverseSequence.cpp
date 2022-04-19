/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Gather.cpp - Lowering Gather Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXReverseSequenceOpLowering : public ConversionPattern {
  ONNXReverseSequenceOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXReverseSequenceOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXReverseSequenceOpAdaptor operandAdaptor(operands);
    ONNXReverseSequenceOp reverseSequenceOp =
        llvm::cast<ONNXReverseSequenceOp>(op);
    auto loc = op->getLoc();

    ONNXReverseSequenceOpShapeHelper shapeHelper(&reverseSequenceOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput());

    // Save axis and rank info.
    int64_t batchAxis = reverseSequenceOp.batch_axis();
    int64_t timeAxis = reverseSequenceOp.time_axis();

    MemRefBoundsIndexCapture dataBounds(operandAdaptor.input());
    int64_t outputRank = shapeHelper.dimsForOutput().size();
    LiteralIndexExpr oneIE(1);

    /*
      The semantic of Reversequence can be expressed in loop as:

      for (vector of dim I on the shape of output tensor) {
        vector of dim Iinput = I;
        Iinput[time_axis()]  = I[time_axis()] < seqence_lens[I[batch_axis()]]?
              sequence_lens[I[batch_axis()]]-I[time_axis()]-1:I[time_axis()];
        output[I] = input[Iinput]
      }

      Obviously, since the conditional check is on time_axis() loop variable,
      the check can be eliminated with loop splitting as long as the batch_axis
      loop is outside.

      I is vector of dim on the shape of output tensor;
      for ( I except I[batch_axis()]) {
        // Reverse
        for (I[time_axis() :  0 to sequence_lens[I[batch_axis]]) {
          Iinput = I;
          Iinput[time_axis()] = sequence_lens[I[batch_axis()]]-I[time_axis()]-1
          output[I] = input[Iinput];
        }
        // Copy
        for (I[time_axis() from sequence_lens[I[batch_axis]] to end) {
          output[I] = input[I];
        }
      }

      This transformation should improve performance on this loop nest itself,
      but may hinder loop fusion with other loop nestes.
      Also further loop fission or loop interchanging can be applied here.
      I chose the simple loop structure
      and believe optimization should be left to compiler, at least for
      non-critical ops.
    */

    // Define loops and iteration trip counts (equivalent to size of output)
    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange loopDef = createKrnl.defineLoops(outputRank);
    SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
    createKrnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.dimsForOutput(),
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          IndexExprScope innerLoopScope(&rewriter, shapeHelper.scope);

          // compute the loop indices for the output
          SmallVector<IndexExpr, 4> outputAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, outputAccessFct);

          // Compute access function for indices[jjs].
          SmallVector<IndexExpr, 4> inputAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, inputAccessFct);
          Value lensVal = createKrnl.loadIE(
              operandAdaptor.sequence_lens(), inputAccessFct[batchAxis]);
          IndexExpr lens = NonAffineIndexExpr(lensVal);
          IndexExpr timeDim = inputAccessFct[timeAxis];
          IndexExpr cond = timeDim < lens;
          IndexExpr inputIndex =
              IndexExpr::select(cond, lens - timeDim - oneIE, timeDim);
          inputAccessFct[timeAxis] = inputIndex;
          Value inputVal =
              createKrnl.loadIE(operandAdaptor.input(), inputAccessFct);

          // Save data into output
          createKrnl.storeIE(inputVal, alloc, outputAccessFct);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXReverseSequenceOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXReverseSequenceOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
