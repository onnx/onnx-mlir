/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Compress.cpp - Lowering Compress Op -----------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Compress Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXCompressOpLowering : public ConversionPattern {

  ONNXCompressOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXCompressOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = ONNXLoc<ONNXCompressOp>(op);
    KrnlBuilder createKrnl(rewriter, loc);
    MemRefBuilder createMemRef(createKrnl);
    ONNXCompressOpAdaptor operandAdaptor(operands);
    ONNXCompressOp compressOp = llvm::dyn_cast<ONNXCompressOp>(op);
    // Create a few constants.
    Value trueVal =
        emitConstantOp(rewriter, loc, rewriter.getIntegerType(1), 1);
    LiteralIndexExpr zero(0), one(1);

    // Get shape, also deliver normalized "axis", -1 if undef.
    ONNXCompressOpShapeHelper shapeHelper(&compressOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed));

    // First compute how many "true" values there are along the condition, as
    // this defines the dynamic dimension pointed to by axis.
    // Create temp memory for summing up the true value and init to zero.
    MemRefType sumType = MemRefType::get({}, rewriter.getIndexType());
    Value sumMemRef = createMemRef.alloca(sumType);
    createKrnl.store(zero.getValue(), sumMemRef);
    // Now create a loop to iterate over all conditions.
    Value condMemRef = operandAdaptor.condition();
    MemRefBoundsIndexCapture condBounds(condMemRef);
    ValueRange loopDef = createKrnl.defineLoops(1);
    createKrnl.iterateIE(loopDef, loopDef, {zero}, {condBounds.getDim(0)},
        [&](KrnlBuilder createKrnl, ValueRange loopInd) {
          MathBuilder createMath(createKrnl);
          // Load the condition
          Value currCond = createKrnl.load(condMemRef, loopInd); // Type i1.
          Value isOn = createMath.eq(currCond, trueVal);         // Compare i1s.
          Value inc = createMath.select(isOn, one.getValue(), zero.getValue());
          Value oldSum = createKrnl.load(sumMemRef);
          Value newSum = createMath.add(oldSum, inc); // Increment by 0 or 1.
          createKrnl.store(newSum, sumMemRef);
        });
    // Now replace questionmark by actual computed size.
    Value sum = createKrnl.load(sumMemRef);
    DimIndexExpr dynDim(sum);
    if (shapeHelper.axis == -1) {
      shapeHelper.dimsForOutput(0)[0] = dynDim;
    } else {
      shapeHelper.dimsForOutput(0)[shapeHelper.axis] = dynDim;
    }
    // Insert an allocation and deallocation for the result of this operation.
    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));

    // Perform the copy depending on the conditions.
    // We will store the current index to write into the output array in
    // indexMemRef. We reuse here the same memref as used to sum the true
    // predicates.
    Value indexMemRef = sumMemRef;
    createKrnl.store(zero.getValue(), indexMemRef);
    // Get input shape.
    Value inputMemRef = operandAdaptor.condition();
    MemRefBoundsIndexCapture inputBounds(inputMemRef);
    int64_t inputRank = inputBounds.getRank();
    SmallVector<IndexExpr, 4> inputLbs(inputRank, zero);
    SmallVector<IndexExpr, 4> inputUbs;
    inputBounds.getSymbolList(inputUbs);
    // Consider the cases.
    if (shapeHelper.axis == -1) {
      // The output is 1D.
      ValueRange inputLoopDef = createKrnl.defineLoops(inputRank);
      createKrnl.iterateIE(inputLoopDef, inputLoopDef, inputLbs,
          inputUbs, [&](KrnlBuilder createKrnl, ValueRange inputLoopInd){
              
          });
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXCompressOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXCompressOpLowering>(ctx);
}
