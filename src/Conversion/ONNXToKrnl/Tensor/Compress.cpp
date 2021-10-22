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
    LiteralIndexExpr zero(0), one(1);

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
    Value writeIndexMemRef = sumMemRef;
    createKrnl.store(zero.getValue(), writeIndexMemRef);
    // Get input shape.
    Value inputMemRef = operandAdaptor.input();
    MemRefBoundsIndexCapture inputBounds(inputMemRef);
    int64_t inputRank = inputBounds.getRank();
    SmallVector<IndexExpr, 4> inputLbs(inputRank, zero);
    SmallVector<IndexExpr, 4> inputUbs;
    inputBounds.getSymbolList(inputUbs);
    printf("hi alex, axis is %d\n", (int)shapeHelper.axis);
    // Consider the cases.
    if (shapeHelper.axis == -1) {
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
      Value readIndexMemRef = createMemRef.alloca(indexMemRefType);
      createKrnl.store(zero.getValue(), readIndexMemRef);

      Value condUb = condBounds.getSymbol(0).getValue();
      ValueRange inputLoopDef = createKrnl.defineLoops(inputRank);
      createKrnl.iterateIE(inputLoopDef, inputLoopDef, inputLbs, inputUbs,
          [&](KrnlBuilder createKrnl, ValueRange inputLoopInd) {
            MathBuilder createMath(createKrnl);
            SCFBuilder createSCF(createKrnl);
            Value readIndex = createKrnl.load(readIndexMemRef);
            Value inBound = createMath.slt(readIndex, condUb);
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
      // for i0, i1,... im
      //   if im < condUB) /* if larger, there are no more true vals */
      //     if cond[im] == true
      //       for im+1..in-1:
      //          val = input[i0...in-1]
      //          out[i0...im-1, writeIndex, im+1...in-1]
      //       writeIndex++

#if 1
      int axis = shapeHelper.axis;
      int outerRank = axis + 1;
      int innerRank = inputRank - outerRank;
      printf("hi alex, axis is %d, outer rank is %d, inner rank is %d\n", axis, outerRank, innerRank);
      Value condUb = condBounds.getSymbol(0).getValue();
      // Divide the loop defs into the outer and inner loop as above
      SmallVector<IndexExpr, 4> outerLbs, outerUbs, innerLbs, innerUbs;
      // Separate here the bounds between outer and inner.
      for (int i = 0; i < outerRank; ++i) {
        outerLbs.emplace_back(inputLbs[i]);
        outerUbs.emplace_back(inputUbs[i]);
      }
      for (int i = outerRank; i < inputRank; ++i) {
        innerLbs.emplace_back(inputLbs[i]);
        innerUbs.emplace_back(inputUbs[i]);
      }
      assert((int)innerLbs.size() == innerRank && "faulty rank calculation");
      ValueRange outerLoopDefs = createKrnl.defineLoops(outerRank);
      createKrnl.iterateIE(outerLoopDefs, outerLoopDefs, outerLbs, outerUbs,
          [&](KrnlBuilder createKrnl, ValueRange outerLoopInd) {
            MathBuilder createMath(createKrnl);
            SCFBuilder createSCF(createKrnl);

            Value readIndex = outerLoopInd[axis]; // Last iter is axis index.
            Value inBound = createMath.slt(readIndex, condUb);

            createSCF.ifThenElse(inBound, [&](SCFBuilder &createSCF) {
              KrnlBuilder createKrnl(createSCF);
              MathBuilder createMath(createSCF);
              Value currCond = createKrnl.load(condMemRef, {readIndex});
              Value copy = createMath.neq(currCond, falseVal);
              createSCF.ifThenElse(copy, [&](SCFBuilder &createSCF) {
                KrnlBuilder createKrnl(createSCF);
                // Now iterate over the inner loops

                ValueRange innerLoopDefs = createKrnl.defineLoops(innerRank);
                createKrnl.iterateIE(innerLoopDefs, innerLoopDefs, innerLbs,
                    innerUbs,
                    [&](KrnlBuilder createKrnl, ValueRange innerLoopInd) {
                      MathBuilder createMath(createKrnl);
                      // Compute access functions for input and output.
                      SmallVector<Value, 4> inputAccessFct, outputAccessFct;
                      for (int i = 0; i < outerRank; ++i) {
                        inputAccessFct.emplace_back(outerLoopInd[i]);
                        outputAccessFct.emplace_back(outerLoopInd[i]);
                      }
                      for (int i = 0; i < innerRank; ++i) {
                        inputAccessFct.emplace_back(innerLoopInd[i]);
                        outputAccessFct.emplace_back(innerLoopInd[i]);
                      }
                      // Only difference for output: write index on axis dim.
                      Value writeIndex = createKrnl.load(writeIndexMemRef);
                      outputAccessFct[axis] = writeIndex;
                      // Now load and copy the result to the output.
                      Value val = createKrnl.load(inputMemRef, inputAccessFct);
                      createKrnl.store(val, alloc, outputAccessFct);
                    });
                // Done with copying, now increment writeIndex
                // Update write index
                Value writeIndex = createKrnl.load(writeIndexMemRef);
                Value one = createMath.constant(indexType, 1);
                Value newWriteIndex = createMath.add(writeIndex, one);
                createKrnl.store(newWriteIndex, writeIndexMemRef);
              }); // If we must copy.
            });   // If we are inbound for tests.
          });

#else
      // Version that expose a krnl lowering bug
      Value condUb = condBounds.getSymbol(0).getValue();
      ValueRange inputLoopDef = createKrnl.defineLoops(inputRank);
      // Divide the loop defs into the outer and inner loop as above
      SmallVector<Value, 4> outerLoopDef, innerLoopDef;
      int axis = shapeHelper.axis;
      for (int i = 0; i <= axis; ++i)
        outerLoopDef.emplace_back(inputLoopDef[i]);
      for (int i = axis + 1; i < inputRank; ++i)
        innerLoopDef.emplace_back(inputLoopDef[i]);

      // This should work, give all of the bounds but request iteration only
      // over the outer loops.
      createKrnl.iterateIE(inputLoopDef, outerLoopDef, inputLbs, inputUbs,
          [&](KrnlBuilder createKrnl, ValueRange outerLoopInd) {
            MathBuilder createMath(createKrnl);
            SCFBuilder createSCF(createKrnl);

            Value readIndex = outerLoopInd[axis]; // Last iter is axis index.
            Value inBound = createMath.slt(readIndex, condUb);

            createSCF.ifThenElse(inBound, [&](SCFBuilder &createSCF) {
              KrnlBuilder createKrnl(createSCF);
              MathBuilder createMath(createSCF);
              Value currCond = createKrnl.load(condMemRef, {readIndex});
              Value copy = createMath.neq(currCond, falseVal);
              createSCF.ifThenElse(copy, [&](SCFBuilder &createSCF) {
                KrnlBuilder createKrnl(createSCF);
                // Now iterate over the inner loops
                createKrnl.iterateIE({}, innerLoopDef, {}, {},
                    [&](KrnlBuilder createKrnl, ValueRange innerLoopInd) {
                      MathBuilder createMath(createKrnl);
                      // Compute access functions for input and output.
                      SmallVector<Value, 4> inputAccessFct, outputAccessFct;
                      for (int i = 0; i <= axis; ++i) {
                        inputAccessFct.emplace_back(outerLoopInd[i]);
                        outputAccessFct.emplace_back(outerLoopInd[i]);
                      }
                      int innerSize = innerLoopInd.size();
                      for (int i = 0; i < innerSize; ++i) {
                        inputAccessFct.emplace_back(innerLoopInd[i]);
                        outputAccessFct.emplace_back(innerLoopInd[i]);
                      }
                      // Only difference for output: write index on axis dim.
                      Value writeIndex = createKrnl.load(writeIndexMemRef);
                      outputAccessFct[axis] = writeIndex;
                      // Now load and copy the result to the output.
                      Value val = createKrnl.load(inputMemRef, inputAccessFct);
                      createKrnl.store(val, alloc, outputAccessFct);
                    });
                // Done with copying, now increment writeIndex
                // Update write index
                Value writeIndex = createKrnl.load(writeIndexMemRef);
                Value one = createMath.constant(indexType, 1);
                Value newWriteIndex = createMath.add(writeIndex, one);
                createKrnl.store(newWriteIndex, writeIndexMemRef);
              }); // If we must copy.
            });   // If we are inbound for tests.
          });
#endif
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXCompressOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXCompressOpLowering>(ctx);
}
