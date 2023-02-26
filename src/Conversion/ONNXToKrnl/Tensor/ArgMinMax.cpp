/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ArgMinMax.cpp - Lowering ArgMin/ArgMax Op ---------------===//
//
// Copyright 2021-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ArgMin/ArgMax Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <typename ARG_OP>
inline Value getCondition(MathBuilder createMath, Value next, Value dstVal);

template <>
inline Value getCondition<ONNXArgMinOp>(
    MathBuilder createMath, Value next, Value dstVal) {
  return createMath.slt(next, dstVal);
}

template <>
inline Value getCondition<ONNXArgMaxOp>(
    MathBuilder createMath, Value next, Value dstVal) {
  return createMath.sgt(next, dstVal);
}

template <typename ARG_OP>
struct ONNXArgMinMaxOpLowering : public ConversionPattern {
  ONNXArgMinMaxOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, ARG_OP::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Location loc = op->getLoc();
    IndexExprScope scope(&rewriter, loc);
    ARG_OP argOp = llvm::cast<ARG_OP>(op);
    typename ARG_OP::Adaptor operandAdaptor(operands);
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXArgMinMaxOpShapeHelper<ARG_OP> shapeHelper(
        op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr outputDims = shapeHelper.getOutputDims();

    // Convert the reduced output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType reducedMemRefType = convertedType.cast<MemRefType>();
    Type reducedElementType = reducedMemRefType.getElementType();
    int64_t reducedRank = reducedMemRefType.getRank();

    // data input
    Value data = operandAdaptor.getData();
    MemRefType dataType = data.getType().cast<MemRefType>();
    int64_t dataRank = dataType.getRank();

    // axis & keepdims attribute
    int64_t axis = argOp.getAxis();
    assert(axis >= -dataRank && axis <= dataRank - 1);
    axis = axis >= 0 ? axis : (dataRank + axis);

    int64_t keepdims = argOp.getKeepdims();
    bool isKeepdims = (keepdims == 1) ? true : false;

    // Get type information
    llvm::SmallVector<int64_t, 1> axes;
    axes.push_back(axis);
    std::map<int64_t, int64_t> outInDimMap =
        getReductionMapping(dataType, llvm::ArrayRef(axes), isKeepdims);

    // Insert alloc and dealloc
    Value alloc = create.mem.alignedAlloc(reducedMemRefType, outputDims);

    // Constant Value
    Value minusOne = create.math.constant(reducedElementType, -1);
    Value zero = create.math.constant(reducedElementType, 0);
    auto zeroIndex = create.math.constantIndex(0);

    // 1. Krnl loops to initialize the result.
    ValueRange initLoopDef = create.krnl.defineLoops(reducedRank);
    SmallVector<IndexExpr, 4> initLbs(reducedRank, LiteralIndexExpr(0));
    create.krnl.iterateIE(initLoopDef, initLoopDef, initLbs, outputDims,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          createKrnl.store(minusOne, alloc, loopInd);
        });

    // 2. Krnl loop to calculate arg min/arg max.
    ValueRange calcLoopDef = create.krnl.defineLoops(dataRank);
    SmallVector<IndexExpr, 4> lbs(dataRank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(data, ubs);
    create.krnl.iterateIE(calcLoopDef, calcLoopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Handle the operation:
          SmallVector<Value, 4> inLoopIVs, outLoopIVs, dstLoopIVs;

          for (int i = 0; i < dataRank; ++i)
            inLoopIVs.push_back(loopInd[i]);

          for (int i = 0; i < reducedRank; ++i) {
            if (outInDimMap.find(i) != outInDimMap.end())
              outLoopIVs.push_back(inLoopIVs[outInDimMap[i]]);
            else
              outLoopIVs.push_back(zeroIndex);
          }

          Value next = createKrnl.load(data, inLoopIVs);
          Value idx = createKrnl.load(alloc, outLoopIVs);

          // if index is less than 0, we should set 0 as initial position
          Value lessThanZero = create.math.slt(idx, zero);
          idx = create.math.select(lessThanZero, zero, idx);

          // induction variables of current min/max value
          for (int i = 0; i < dataRank; ++i) {
            if (i != axis)
              dstLoopIVs.push_back(loopInd[i]);
            else
              dstLoopIVs.push_back(rewriter.create<arith::IndexCastOp>(
                  loc, rewriter.getIndexType(), idx));
          }
          Value dstVal = createKrnl.load(data, dstLoopIVs);

          // if next value is smaller/larger than current value, update index
          Value newDstVal = getCondition<ARG_OP>(create.math, next, dstVal);
          Value pos =
              create.math.cast(rewriter.getIntegerType(64), inLoopIVs[axis]);
          idx = create.math.select(newDstVal, pos, idx);
          createKrnl.store(idx, alloc, outLoopIVs);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXArgMinMaxOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXArgMinMaxOpLowering<mlir::ONNXArgMinOp>>(
      typeConverter, ctx);
  patterns.insert<ONNXArgMinMaxOpLowering<mlir::ONNXArgMaxOp>>(
      typeConverter, ctx);
}

} // namespace onnx_mlir
