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
    MathBuilder createMath(createKrnl);
    ONNXCompressOpAdaptor operandAdaptor(operands);
    ONNXCompressOp compressOp = llvm::dyn_cast<ONNXCompressOp>(op);

    // Get shape, also deliver normalized "axis", -1 if undef.
    ONNXCompressOpShapeHelper shapeHelper(&compressOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed));

    // Create a few constants.
    auto bitType = rewriter.getIntegerType(1);
    Value falseVal = createMath.constant(bitType, 0);
    Value trueVal = createMath.constant(bitType, 1);
    LiteralIndexExpr zero(0), one(1);
    int axis = shapeHelper.axis;

    // First compute how many "true" values there are along the condition, as
    // this defines the dynamic dimension pointed to by axis.
    // Create temp memory for summing up the true value and init to zero.
    Type indexType = rewriter.getIndexType();
    MemRefType indexMemRefType = MemRefType::get({}, indexType);
    Value sumMemRef = createMemRef.alloca(indexMemRefType);
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
          Value isOn = createMath.neq(currCond, falseVal);       // Compare i1s.
          Value inc = createMath.select(isOn, one.getValue(), zero.getValue());
          Value oldSum = createKrnl.load(sumMemRef);
          Value newSum = createMath.add(oldSum, inc); // Increment by 0 or 1.
          createKrnl.store(newSum, sumMemRef);
        });
    // Now replace questionmark by actual computed size.
    Value sum = createKrnl.load(sumMemRef);
    DimIndexExpr dynDim(sum);
    if (axis == -1) {
      shapeHelper.dimsForOutput(0)[0] = dynDim;
    } else {
      shapeHelper.dimsForOutput(0)[axis] = dynDim;
    }

    // Insert an allocation and deallocation for the result of this operation.
    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));

    // Perform the copy depending on the conditions.
    // We will store the current index to write into the output array in
    // indexMemRef. We reuse here the same memref as used to sum the true
    // predicates.
    Value writeIndexMemRef = sumMemRef;
    createKrnl.store(zero.getValue(), writeIndexMemRef);
    // Get input shape.
    Value inputMemRef = operandAdaptor.input();
    MemRefBoundsIndexCapture inputBounds(inputMemRef);
    int64_t inputRank = inputBounds.getRank();
    SmallVector<IndexExpr, 4> inputLbs(inputRank, zero);
    SmallVector<IndexExpr, 4> inputUbs;
    inputBounds.getSymbolList(inputUbs);

    // Consider the cases.
    if (axis == -1) {
      // We iterate over the original loops, and in the innerblock we test for
      // the condition. The output is 1D.
      //
      // readIndex = writeIndex = 0;
      // for(i, j, k...)
      //    if (readIndex<condUB) /* non affine condition */
      //       if (cond[readIndex] == true) /* non affine condition */
      //          output[writeIndex] = input[i, j, k...]
      //          writeIndex++
      //       readIndex++
      //    else break; /* not possible with affine loops... ignore */
      //
      // WriteIndex is already init to zero, create and zero readIndex

      // Try to see if we can guarantee that there are enough bits in the
      // condition tensor.
      IndexExpr condSize = condBounds.getSymbol(0);
      Value condUb = condSize.getValue();
      bool skipCond = false;
      if (condSize.isLiteral()) {
        // Make sure that the total size of the input is known at compile time
        // and is smaller that the size of the condition array.
        skipCond = true;
        int64_t inputTotSize = 1;
        for (int i = 0; i < inputRank && skipCond; ++i) {
          if (!inputUbs[i].isLiteral()) {
            skipCond = false; // Runtime input size, cannot skip test.
          } else {
            inputTotSize *= inputUbs[i].getLiteral();
            if (inputTotSize > condSize.getLiteral())
              skipCond = false; // Cond tensor too small, cannot skip test.
          }
        }
      }

      Value readIndexMemRef = createMemRef.alloca(indexMemRefType);
      createKrnl.store(zero.getValue(), readIndexMemRef);

      ValueRange inputLoopDef = createKrnl.defineLoops(inputRank);
      createKrnl.iterateIE(inputLoopDef, inputLoopDef, inputLbs, inputUbs,
          [&](KrnlBuilder createKrnl, ValueRange inputLoopInd) {
            MathBuilder createMath(createKrnl);
            SCFBuilder createSCF(createKrnl);
            Value readIndex = createKrnl.load(readIndexMemRef);
            Value inBound = trueVal;
            if (!skipCond)
              inBound = createMath.slt(readIndex, condUb);
            createSCF.ifThenElse(inBound, [&](SCFBuilder &createSCF) {
              KrnlBuilder createKrnl(createSCF);
              MathBuilder createMath(createSCF);
              Value currCond = createKrnl.load(condMemRef, {readIndex});
              Value copy = createMath.neq(currCond, falseVal);
              createSCF.ifThenElse(copy, [&](SCFBuilder &createSCF) {
                KrnlBuilder createKrnl(createSCF);
                MathBuilder createMath(createSCF);
                Value val = createKrnl.load(inputMemRef, inputLoopInd);
                // Copy to output.
                Value writeIndex = createKrnl.load(writeIndexMemRef);
                createKrnl.store(val, alloc, {writeIndex});
                // Update write index
                Value one = createMath.constant(indexType, 1);
                Value newWriteIndex = createMath.add(writeIndex, one);
                createKrnl.store(newWriteIndex, writeIndexMemRef);
              });
              // Update read index
              Value one = createMath.constant(indexType, 1);
              Value newReadIndex = createMath.add(readIndex, one);
              createKrnl.store(newReadIndex, readIndexMemRef);
            });
          });
    } else {
      // Handle case where output is multi-dimensional.
      //
      // input has rank n, axis is m in 0..n-1
      //
      // writeIndex = 0
      // for im
      //   if im < condUB) /* if larger, there are no more true vals */
      //     if cond[im] == true
      //       for i1... im-1, im+1... in
      //          val = input[i0...im-1, im, im+1...in-1]
      //          out[i0...im-1, writeIndex, im+1...in-1]
      //       writeIndex++

      // Try to see if we can guarantee that there are enough bits in the
      // condition tensor.
      IndexExpr condSize = condBounds.getSymbol(0);
      Value condUb = condSize.getValue();
      bool skipCond = false;
      IndexExpr condTest = (condSize >= inputUbs[axis]);
      if (condTest.isLiteral() && condTest.getLiteral() != 0) {
        // Were able to prove that the ub test is always true
        skipCond = true;
      }
      int innerRank = inputRank - 1;
      // Divide the loop defs into the outer and inner loop as above
      SmallVector<IndexExpr, 4> innerLbs, innerUbs;
      // Separate here the bounds between outer and inner.
      for (int i = 0; i < inputRank; ++i) {
        if (i == axis)
          continue;
        innerLbs.emplace_back(inputLbs[i]);
        innerUbs.emplace_back(inputUbs[i]);
      }
      ValueRange axisLoopDef = createKrnl.defineLoops(1);
      createKrnl.iterateIE(axisLoopDef, axisLoopDef, {inputLbs[axis]},
          {inputUbs[axis]},
          [&](KrnlBuilder createKrnl, ValueRange axisLoopInd) {
            MathBuilder createMath(createKrnl);
            SCFBuilder createSCF(createKrnl);
            // Compute the test if we have enough condition value for current
            // index.
            Value readIndex = axisLoopInd[0];
            Value inBound = trueVal;
            if (!skipCond)
              inBound = createMath.slt(readIndex, condUb);
            createSCF.ifThenElse(inBound, [&](SCFBuilder &createSCF) {
              KrnlBuilder createKrnl(createSCF);
              MathBuilder createMath(createSCF);
              Value currCond = createKrnl.load(condMemRef, {readIndex});
              Value copy = createMath.neq(currCond, falseVal);
              createSCF.ifThenElse(copy, [&](SCFBuilder &createSCF) {
                KrnlBuilder createKrnl(createSCF);
                // Load the write index.
                Value writeIndex = createKrnl.load(writeIndexMemRef);
                // Now iterate over the inner loops
                ValueRange innerLoopDefs = createKrnl.defineLoops(innerRank);
                createKrnl.iterateIE(innerLoopDefs, innerLoopDefs, innerLbs,
                    innerUbs,
                    [&](KrnlBuilder createKrnl, ValueRange innerLoopInd) {
                      MathBuilder createMath(createKrnl);
                      // Compute access functions for input and output.
                      SmallVector<Value, 4> inputAccessFct, outputAccessFct;
                      int skip = 0;
                      for (int i = 0; i < inputRank; ++i) {
                        if (i == axis) {
                          inputAccessFct.emplace_back(readIndex);
                          outputAccessFct.emplace_back(writeIndex);
                          skip = 1;
                        } else {
                          inputAccessFct.emplace_back(innerLoopInd[i - skip]);
                          outputAccessFct.emplace_back(innerLoopInd[i - skip]);
                        }
                      }
                      // Now load and copy the result to the output.
                      Value val = createKrnl.load(inputMemRef, inputAccessFct);
                      createKrnl.store(val, alloc, outputAccessFct);
                    });
                // Done with copying, now increment writeIndex
                // Value writeIndex = createKrnl.load(writeIndexMemRef);
                Value one = createMath.constant(indexType, 1);
                Value newWriteIndex = createMath.add(writeIndex, one);
                createKrnl.store(newWriteIndex, writeIndexMemRef);
              }); // If we must copy.
            });   // If we are inbound for tests.
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
