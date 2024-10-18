/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Compress.cpp - Lowering Compress Op -----------------===//
//
// Copyright 2021-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Compress Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXCompressOpLowering : public OpConversionPattern<ONNXCompressOp> {

  ONNXCompressOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXCompressOp compressOp,
      ONNXCompressOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = compressOp.getOperation();
    Location loc = ONNXLoc<ONNXCompressOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Get shape, also deliver normalized "axis", -1 if undef.
    ONNXCompressOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Get input shape.
    Value inputMemRef = adaptor.getInput();
    int64_t inputRank = create.krnlIE.getShapedTypeRank(inputMemRef);
    std::optional<int64_t> axis = compressOp.getAxis();

    // Create a few constants.
    auto bitType = rewriter.getIntegerType(1);
    Value falseVal = create.math.constant(bitType, 0);
    Value trueVal = create.math.constant(bitType, 1);
    LiteralIndexExpr zeroIE(0), oneIE(1);

    // First compute how many "true" values there are along the condition, as
    // this defines the dynamic dimension pointed to by axis.
    // Create temp memory for summing up the true value and init to zero.
    Type indexType = rewriter.getIndexType();
    MemRefType indexMemRefType = MemRefType::get({}, indexType);
    // Scalar, ok to use alloca.
    Value sumMemRef = create.mem.alloca(indexMemRefType);
    create.krnl.store(zeroIE.getValue(), sumMemRef);
    // Now create a loop to iterate over all conditions.
    Value condMemRef = adaptor.getCondition();
    IndexExpr condShapeFirstRank = create.krnlIE.getShapeAsDim(condMemRef, 0);
    create.krnl.forLoopIE(zeroIE, condShapeFirstRank, /*step*/ 1, /*par*/ false,
        [&](const KrnlBuilder createKrnl, ValueRange loopInd) {
          MathBuilder createMath(createKrnl);
          // Load the condition
          Value currCond = createKrnl.load(condMemRef, loopInd); // Type i1.
          Value isOn = createMath.neq(currCond, falseVal);       // Compare i1s.
          Value inc =
              createMath.select(isOn, oneIE.getValue(), zeroIE.getValue());
          Value oldSum = createKrnl.load(sumMemRef);
          Value newSum = createMath.add(oldSum, inc); // Increment by 0 or 1.
          createKrnl.store(newSum, sumMemRef);
        });

    // Now replace questionmark by actual computed size.
    Value sum = create.krnl.load(sumMemRef);
    DimIndexExpr dynDim(sum);
    if (!axis.has_value()) {
      shapeHelper.getOutputDims()[0] = dynDim;
    } else {
      const int64_t axisValue =
          (axis.value() >= 0) ? axis.value() : axis.value() + inputRank;
      shapeHelper.getOutputDims()[axisValue] = dynDim;
    }

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc =
        create.mem.alignedAlloc(memRefType, shapeHelper.getOutputDims());

    // Perform the copy depending on the conditions.
    // We will store the current index to write into the output array in
    // indexMemRef. We reuse here the same memref as used to sum the true
    // predicates.
    Value writeIndexMemRef = sumMemRef;
    create.krnl.store(zeroIE.getValue(), writeIndexMemRef);

    SmallVector<IndexExpr, 4> inputLbs(inputRank, zeroIE);
    SmallVector<IndexExpr, 4> inputUbs;
    create.krnlIE.getShapeAsSymbols(inputMemRef, inputUbs);

    // Consider the cases.
    if (!axis.has_value()) {
      // We iterate over the original loops, and in the inner block we test for
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
      IndexExpr condSize = create.krnlIE.getShapeAsSymbol(condMemRef, 0);
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

      // Scalar, ok to use alloca.
      Value readIndexMemRef = create.mem.alloca(indexMemRefType);
      create.krnl.store(zeroIE.getValue(), readIndexMemRef);

      ValueRange inputLoopDef = create.krnl.defineLoops(inputRank);
      create.krnl.iterateIE(inputLoopDef, inputLoopDef, inputLbs, inputUbs,
          [&](const KrnlBuilder createKrnl, ValueRange inputLoopInd) {
            MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder> create(
                createKrnl);
            Value readIndex = create.krnl.load(readIndexMemRef);
            Value inBound = trueVal;
            if (!skipCond)
              inBound = create.math.slt(readIndex, condUb);
            create.scf.ifThenElse(inBound, [&](const SCFBuilder &createSCF) {
              MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder> create(
                  createSCF);
              Value currCond = create.krnl.load(condMemRef, {readIndex});
              Value copy = create.math.neq(currCond, falseVal);
              create.scf.ifThenElse(copy, [&](const SCFBuilder &createSCF) {
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createSCF);
                Value val = create.krnl.load(inputMemRef, inputLoopInd);
                // Copy to output.
                Value writeIndex = create.krnl.load(writeIndexMemRef);
                create.krnl.store(val, alloc, {writeIndex});
                // Update write index
                Value one = create.math.constant(indexType, 1);
                Value newWriteIndex = create.math.add(writeIndex, one);
                create.krnl.store(newWriteIndex, writeIndexMemRef);
              });
              // Update read index
              Value one = create.math.constant(indexType, 1);
              Value newReadIndex = create.math.add(readIndex, one);
              create.krnl.store(newReadIndex, readIndexMemRef);
            });
          });
    } else {
      assert(axis.has_value() && "Expecting axis to have a value");
      const int64_t axisValue =
          (axis.value() >= 0) ? axis.value() : axis.value() + inputRank;

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
      IndexExpr condSize = create.krnlIE.getShapeAsSymbol(condMemRef, 0);
      Value condUb = condSize.getValue();
      bool skipCond = false;
      IndexExpr condTest = (condSize >= inputUbs[axisValue]);
      if (condTest.isLiteral() && condTest.getLiteral() != 0) {
        // Were able to prove that the ub test is always true
        skipCond = true;
      }
      int innerRank = inputRank - 1;
      // Divide the loop defs into the outer and inner loop as above
      SmallVector<IndexExpr, 4> innerLbs, innerUbs;
      // Separate here the bounds between outer and inner.
      for (int i = 0; i < inputRank; ++i) {
        if (i == axisValue)
          continue;
        innerLbs.emplace_back(inputLbs[i]);
        innerUbs.emplace_back(inputUbs[i]);
      }
      create.krnl.forLoopIE(inputLbs[axisValue], inputUbs[axisValue],
          /*step*/ 1, /*par*/ false,
          [&](const KrnlBuilder createKrnl, ValueRange axisLoopInd) {
            MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder> create(
                createKrnl);
            // Compute the test if we have enough condition value for current
            // index.
            Value readIndex = axisLoopInd[0];
            Value inBound = trueVal;
            if (!skipCond)
              inBound = create.math.slt(readIndex, condUb);
            create.scf.ifThenElse(inBound, [&](const SCFBuilder &createSCF) {
              MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder> create(
                  createSCF);
              Value currCond = create.krnl.load(condMemRef, {readIndex});
              Value copy = create.math.neq(currCond, falseVal);
              create.scf.ifThenElse(copy, [&](const SCFBuilder &createSCF) {
                KrnlBuilder createKrnl(createSCF);
                // Load the write index.
                Value writeIndex = createKrnl.load(writeIndexMemRef);
                // Now iterate over the inner loops
                ValueRange innerLoopDefs = createKrnl.defineLoops(innerRank);
                createKrnl.iterateIE(innerLoopDefs, innerLoopDefs, innerLbs,
                    innerUbs,
                    [&](const KrnlBuilder createKrnl, ValueRange innerLoopInd) {
                      MathBuilder createMath(createKrnl);
                      // Compute access functions for input and output.
                      SmallVector<Value, 4> inputAccessFct, outputAccessFct;
                      int skip = 0;
                      for (int i = 0; i < inputRank; ++i) {
                        if (i == axisValue) {
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
                Value one = create.math.constant(indexType, 1);
                Value newWriteIndex = create.math.add(writeIndex, one);
                create.krnl.store(newWriteIndex, writeIndexMemRef);
              }); // If we must copy.
            });   // If we are inbound for tests.
          });
    }
    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXCompressOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXCompressOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
