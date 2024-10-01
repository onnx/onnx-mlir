/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ReverseSequence.cpp - Lowering ReverseSequence Op-----------=== //
//
// Copyright 2020-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXReverseSequenceOpLowering
    : public OpConversionPattern<ONNXReverseSequenceOp> {
  ONNXReverseSequenceOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXReverseSequenceOp reverseSequenceOp,
      ONNXReverseSequenceOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = reverseSequenceOp.getOperation();
    Location loc = ONNXLoc<ONNXReverseSequenceOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXReverseSequenceOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Save axis and rank info.
    int64_t batchAxis = reverseSequenceOp.getBatchAxis();
    int64_t timeAxis = reverseSequenceOp.getTimeAxis();

    int64_t outputRank = shapeHelper.getOutputDims().size();
    LiteralIndexExpr oneIE(1);

    /*
      The semantic of ReverseSequence can be expressed in loop as:

      for (vector of dim I on the shape of output tensor) {
        vector of dim Iinput = I;
        Iinput[time_axis()]  = I[time_axis()] < sequence_lens[I[batch_axis()]]?
              sequence_lens[I[batch_axis()]]-I[time_axis()]-1:I[time_axis()];
        output[I] = input[Iinput]
      }

      Obviously, since the conditional check is on getTimeAxis() loop variable,
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
      but may hinder loop fusion with other loop nests.
      Also further loop fission or loop interchanging can be applied here.
      I chose the simple loop structure
      and believe optimization should be left to compiler, at least for
      non-critical ops.
    */

    // Define loops and iteration trip counts (equivalent to size of output)
    ValueRange loopDef = create.krnl.defineLoops(outputRank);
    SmallVector<IndexExpr, 4> lbs(outputRank, LitIE(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.getOutputDims(),
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          IndexExprScope innerLoopScope(&rewriter, shapeHelper.getScope());

          // compute the loop indices for the output
          SmallVector<IndexExpr, 4> outputAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, outputAccessFct);

          // Compute access function for indices[jjs].
          SmallVector<IndexExpr, 4> inputAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, inputAccessFct);
          Value lensVal = createKrnl.loadIE(
              adaptor.getSequenceLens(), inputAccessFct[batchAxis]);
          IndexExpr lens = NonAffineIndexExpr(lensVal);
          IndexExpr timeDim = inputAccessFct[timeAxis];
          IndexExpr cond = timeDim < lens;
          IndexExpr inputIndex =
              IndexExpr::select(cond, lens - timeDim - oneIE, timeDim);
          inputAccessFct[timeAxis] = inputIndex;
          Value inputVal =
              createKrnl.loadIE(adaptor.getInput(), inputAccessFct);

          // Save data into output
          createKrnl.storeIE(inputVal, alloc, outputAccessFct);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXReverseSequenceOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXReverseSequenceOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
