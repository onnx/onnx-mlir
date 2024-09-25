/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- OneHot.cpp - Lowering OneHot Op -------------------===//
//
// Copyright 2021-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX OneHot Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXOneHotOpLowering : public OpConversionPattern<ONNXOneHotOp> {
  ONNXOneHotOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXOneHotOp onehotOp,
      ONNXOneHotOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = onehotOp.getOperation();
    Location loc = ONNXLoc<ONNXOneHotOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value indices = adaptor.getIndices();
    Value values = adaptor.getValues();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXOneHotOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    int64_t axis = shapeHelper.axis;

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Load off/on vals found in values memref.
    LiteralIndexExpr minusOneIE(-1), zeroIE(0), oneIE(1);
    Value offVal = create.krnl.loadIE(values, zeroIE);
    Value onVal = create.krnl.loadIE(values, oneIE);

    // Iterate over all of the inputs.
    int64_t indicesRank = create.krnlIE.getShapedTypeRank(indices);
    SmallVector<IndexExpr, 4> indicesLbs(indicesRank, zeroIE);
    SmallVector<IndexExpr, 4> indicesUbs;
    create.krnlIE.getShapeAsDims(indices, indicesUbs);
    ValueRange indicesLoopDef = create.krnl.defineLoops(indicesRank);
    create.krnl.iterateIE(indicesLoopDef, indicesLoopDef, indicesLbs,
        indicesUbs,
        [&](const KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
          // Loop for all input values.
          MathBuilder createMath(createKrnl);
          // Input val is allowed to be any integer/float. Read and convert to
          // index type.
          Value inputVal = createKrnl.load(indices, indicesLoopInd);
          Value inputIndexVal = createMath.castToIndex(inputVal);
          IndexExprScope innerScope(createKrnl, shapeHelper.getScope());
          NonAffineIndexExpr input(inputIndexVal);
          SymbolIndexExpr depth(shapeHelper.depth);
          // Because valid input is from [-depth...depth-1], we must add depth
          // to input values that are negative. This will define inputIndex.
          IndexExpr inputNegVal = input + depth;
          IndexExpr isNeg = input < zeroIE;
          IndexExpr inputIndex = IndexExpr::select(isNeg, inputNegVal, input);
          // Now compute in inputIndex is still out of bound, in which case all
          // values are off.
          IndexExpr isTooSmall = inputIndex < zeroIE;
          IndexExpr isTooBig = inputIndex >= depth;
          IndexExpr outOfBound = isTooSmall | isTooBig;
          // Define here the index that has the on Value. If out of bound, put
          // -1 here as this value will never occur.
          IndexExpr onValueIndex =
              IndexExpr::select(outOfBound, minusOneIE, inputIndex);
          Value onValueIndexVal = onValueIndex.getValue();
          // Now we have the index that is on, iterate over the depth values
          // along axis, and set the right one to the value on.
          createKrnl.forLoopIE(zeroIE, depth, /*step*/ 1, /*par*/ false,
              [&](const KrnlBuilder createBuilder, ValueRange depthLoopInd) {
                MathBuilder createMath(createKrnl);
                Value onCond = createMath.eq(depthLoopInd[0], onValueIndexVal);
                Value res = createMath.select(onCond, onVal, offVal);
                // Output access function is input indices with depth index
                // spliced in the axis location.
                SmallVector<Value, 4> outputAccessFct;
                int64_t dec = 0;
                for (int64_t i = 0; i < indicesRank + 1; ++i) {
                  if (i == axis) {
                    outputAccessFct.emplace_back(depthLoopInd[0]);
                    dec = 1;
                  } else {
                    outputAccessFct.emplace_back(indicesLoopInd[i - dec]);
                  }
                }
                createKrnl.store(res, alloc, outputAccessFct);
              });
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXOneHotOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXOneHotOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
