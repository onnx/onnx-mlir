/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ArgMinMax.cpp - Lowering ArgMin/ArgMax Op ---------------===//
//
// Copyright 2021-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ArgMin/ArgMax Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <typename ArgOp>
inline Value getCondition(MultiDialectBuilder<KrnlBuilder, MathBuilder> create,
    Value next, Value dstVal);

template <>
inline Value getCondition<ONNXArgMinOp>(
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create, Value next,
    Value dstVal) {
  return create.math.slt(next, dstVal);
}

template <>
inline Value getCondition<ONNXArgMaxOp>(
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create, Value next,
    Value dstVal) {
  return create.math.sgt(next, dstVal);
}

template <typename ArgOp, typename OpShapeHelper>
struct ONNXArgMinMaxOpLowering : public ConversionPattern {
  ONNXArgMinMaxOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, ArgOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    auto loc = op->getLoc();
    IndexExprScope scope(&rewriter, loc);
    ArgOp argOp = llvm::cast<ArgOp>(op);

    typename ArgOp::Adaptor operandAdaptor(operands);
    OpShapeHelper shapeHelper(&argOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");
    DimsExpr outputDims = shapeHelper.dimsForOutput();

    // Convert the reduced output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType reducedMemRefType = convertedType.cast<MemRefType>();
    Type reducedElementType = reducedMemRefType.getElementType();
    int64_t reducedRank = reducedMemRefType.getRank();

    // data input
    Value data = operandAdaptor.data();
    MemRefType dataType = data.getType().cast<MemRefType>();
    int64_t dataRank = dataType.getRank();

    // axis & keepdims attribute
    int64_t axis = argOp.axis();
    assert(axis >= -dataRank && axis <= dataRank - 1);
    axis = axis >= 0 ? axis : (dataRank + axis);

    int64_t keepdims = argOp.keepdims();
    bool isKeepdims = (keepdims == 1) ? true : false;

    // Get type information
    llvm::SmallVector<int64_t, 1> axes;
    axes.push_back(axis);
    std::map<int64_t, int64_t> outInDimMap =
        getReductionMapping(dataType, llvm::makeArrayRef(axes), isKeepdims);

    // Insert alloc and dealloc
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, reducedMemRefType, loc, outputDims);

    // Constant Value
    MathBuilder createMath(rewriter, loc);
    Value minusOne = createMath.constant(reducedElementType, -1);
    Value zero = createMath.constant(reducedElementType, 0);
    auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    KrnlBuilder createKrnl(rewriter, loc);

    // 1. Krnl loops to initialize the result.
    ValueRange initLoopDef = createKrnl.defineLoops(reducedRank);
    SmallVector<IndexExpr, 4> initLbs(reducedRank, LiteralIndexExpr(0));
    createKrnl.iterateIE(initLoopDef, initLoopDef, initLbs, outputDims,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          createKrnl.store(minusOne, alloc, loopInd);
        });

    // 2. Krnl loop to calculate argmin/argmax.
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
    ValueRange calcLoopDef = createKrnl.defineLoops(dataRank);
    SmallVector<IndexExpr, 4> lbs(dataRank, LiteralIndexExpr(0));
    MemRefBoundsIndexCapture dataBounds(data);
    SmallVector<IndexExpr, 4> ubs;
    dataBounds.getDimList(ubs);
    createKrnl.iterateIE(calcLoopDef, calcLoopDef, lbs, ubs,
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
          Value newDstVal = getCondition<ArgOp>(create, next, dstVal);
          Value pos = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIntegerType(64), inLoopIVs[axis]);
          idx = create.math.select(newDstVal, pos, idx);
          createKrnl.store(idx, alloc, outLoopIVs);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXArgMinMaxOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<
      ONNXArgMinMaxOpLowering<mlir::ONNXArgMinOp, ONNXArgMinOpShapeHelper>>(
      typeConverter, ctx);
  patterns.insert<
      ONNXArgMinMaxOpLowering<mlir::ONNXArgMaxOp, ONNXArgMaxOpShapeHelper>>(
      typeConverter, ctx);
}

} // namespace onnx_mlir
