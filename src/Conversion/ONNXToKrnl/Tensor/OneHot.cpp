/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- OneHot.cpp - Lowering OneHot Op -------------------===//
//
// Copyright 2021-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX OneHot Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXOneHotOpLowering : public ConversionPattern {
  ONNXOneHotOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXOneHotOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXOneHotOpAdaptor operandAdaptor(operands);
    ONNXOneHotOp oneHotOp = llvm::cast<ONNXOneHotOp>(op);
    Location loc = op->getLoc();

    Value indices = operandAdaptor.indices();
    Value values = operandAdaptor.values();

    ONNXOneHotOpShapeHelper shapeHelper(&oneHotOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");
    int64_t axis = shapeHelper.axis;

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput());

    // Load off/on vals found in values memref.
    KrnlBuilder createKrnl(rewriter, loc);
    LiteralIndexExpr minusOneIE(-1), zeroIE(0), oneIE(1);
    Value offVal = createKrnl.loadIE(values, zeroIE);
    Value onVal = createKrnl.loadIE(values, oneIE);

    // Iterate over all of the inputs.
    MemRefBoundsIndexCapture indicesBounds(indices);
    int64_t indicesRank = indicesBounds.getRank();
    SmallVector<IndexExpr, 4> indicesLbs(indicesRank, zeroIE);
    SmallVector<IndexExpr, 4> indicesUbs;
    indicesBounds.getDimList(indicesUbs);
    ValueRange indicesLoopDef = createKrnl.defineLoops(indicesRank);
    createKrnl.iterateIE(indicesLoopDef, indicesLoopDef, indicesLbs, indicesUbs,
        [&](KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
          // Loop for all input values.
          MathBuilder createMath(createKrnl);
          // Input val is allowed to be any integer/float. Read and convert to
          // index type.
          Value inputVal = createKrnl.load(indices, indicesLoopInd);
          Value inputIndexVal = createMath.castToIndex(inputVal);
          IndexExprScope innerScope(createKrnl, shapeHelper.scope);
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
          ValueRange depthLoopDef = createKrnl.defineLoops(1);
          createKrnl.iterateIE(depthLoopDef, depthLoopDef, {zeroIE}, {depth},
              [&](KrnlBuilder createBuilder, ValueRange depthLoopInd) {
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
    return success();
  }
};

void populateLoweringONNXOneHotOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXOneHotOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
