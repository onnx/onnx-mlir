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
    ONNXCompressOpAdaptor operandAdaptor(operands);
    ONNXCompressOp compressOp = llvm::dyn_cast<ONNXCompressOp>(op);
    printf("hi alex 0\n");

    // Get shape, also deliver normalized "axis", -1 if undef.
    ONNXCompressOpShapeHelper shapeHelper(&compressOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    printf("hi alex 0.1\n");
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed));
    printf("hi alex 0.2\n");

    // First compute how many "true" values there are along the condition, as
    // this defines the dynamic dimension pointed to by axis.
    Value condMemRef = operandAdaptor.condition();
    KrnlBuilder createKrnl(rewriter, loc);
    MemRefBuilder createMemRef(createKrnl);
    // Create a few constants.
    Value trueVal =
        emitConstantOp(rewriter, loc, rewriter.getIntegerType(1), 1);
    LiteralIndexExpr zero(0), one(1);
    // Create temp memory for summing up the true value and init to zero.
    MemRefType sumType = MemRefType::get({}, rewriter.getIndexType());
    Value sumMemRef = createMemRef.alloca(sumType);
    createKrnl.store(zero.getValue(), sumMemRef);
    // Now create a loop to iterate over all conditions.
    MemRefBoundsIndexCapture condBounds(condMemRef);
    ValueRange loopDef = createKrnl.defineLoops(1);
    printf("hi alex 1\n");
    createKrnl.iterateIE(loopDef, loopDef, {zero}, {condBounds.getDim(0)},
        [&](KrnlBuilder createKrnl, ValueRange loopInd) {
          MathBuilder createMath(createKrnl);
          // Load the condition
          Value currCond = createKrnl.load(condMemRef, loopInd); // Type i1.
          Value isOn = createMath.eq(currCond, trueVal);         // Compare i1s.
          Value inc = createMath.select(
              isOn, one.getValue(), zero.getValue()); // Inc is set to 0 or 1.
          Value oldSum = createKrnl.load(sumMemRef);
          Value newSum = createMath.add(oldSum, inc);
          createKrnl.store(newSum, sumMemRef);
        });
    // Now replace questionmark by actual computed size.
    printf("hi alex 2\n");
    Value sum = createKrnl.load(sumMemRef);
    DimIndexExpr dynDim(sum);
    if (shapeHelper.axis == -1) {
      shapeHelper.dimsForOutput(0)[0] = dynDim;
    } else {
      shapeHelper.dimsForOutput(0)[shapeHelper.axis] = dynDim;
    }
    printf("hi alex 3\n");
    // Insert an allocation and deallocation for the result of this operation.
    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));
    // Perform the copy depending on the conditions.

    rewriter.replaceOp(op, alloc);
    printf("hi alex 4\n");
    return success();
  }
};

void populateLoweringONNXCompressOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXCompressOpLowering>(ctx);
}
