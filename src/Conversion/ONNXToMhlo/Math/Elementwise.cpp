/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir-hlo/utils/broadcast_utils.h"
#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <>
struct MhloDialectOp<ONNXAddOp> {
  using Op = mhlo::AddOp;
};

template <>
struct MhloDialectOp<ONNXSubOp> {
  using Op = mhlo::SubOp;
};

template <>
struct MhloDialectOp<ONNXDivOp> {
  using Op = mhlo::DivOp;
};

template <>
struct MhloDialectOp<ONNXExpOp> {
  using Op = mhlo::ExpOp;
};

namespace {

// Element-wise unary ops lowering to Mhlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLoweringToMhlo : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseUnaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(StringAttr::get(op->getContext(),
                                    ElementwiseUnaryOp::getOperationName()),
        op->getLoc());
    Value mhloOp = rewriter.create<MhloOp<ElementwiseUnaryOp>>(
        loc, op->getResultTypes(), op->getOperands());
    rewriter.replaceOp(op, mhloOp);
    return success();
  }
};

template <>
struct ONNXElementwiseUnaryOpLoweringToMhlo<ONNXReluOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(ONNXReluOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(
        StringAttr::get(op->getContext(), ONNXReluOp::getOperationName()),
        op->getLoc());
    ONNXReluOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value inp = adaptor.X();
    ShapedType inpType = inp.getType().dyn_cast_or_null<ShapedType>();
    if (inpType == nullptr)
      return failure();
    Type resultType = *op->result_type_begin();
    Value broadcastedZero = getShapedZero(loc, rewriter, inpType, inp, resultType);
    Value resultOp = rewriter.create<mhlo::MaxOp>(loc, resultType, inp, broadcastedZero);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

// Element-wise variadic ops lowering to Mhlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLoweringToMhlo : public ConversionPattern {
  ONNXElementwiseVariadicOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseVariadicOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(StringAttr::get(op->getContext(),
                                    ElementwiseVariadicOp::getOperationName()),
        op->getLoc());

    ONNXGenericOpBroadcastedShapeHelper shapeHelper(op);
    DimsExpr empty;
    LogicalResult shapecomputed = shapeHelper.computeShape(operands, empty);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    int64_t outputRank = shapeHelper.outputRank;
    Type outputType = *op->result_type_begin();
    ShapedType outputShapedType = outputType.dyn_cast<ShapedType>();
    if (outputShapedType == nullptr)
      return failure();
    Type elementType = outputShapedType.getElementType();
    RankedTensorType broadcastedOutputType =
        RankedTensorType::get(outputShapedType.getShape(), elementType);

    Value resultExtents = mlir::hlo::ComputeNaryElementwiseBroadcastingResultExtents(
        loc, op->getOperands(), rewriter);
    llvm::SmallVector<Value, 4> broadcastedOperands;
    for (Value operand : op->getOperands()) {
      RankedTensorType operandType =
          operand.getType().dyn_cast<RankedTensorType>();
      if (operandType == nullptr)
        return failure();
      SmallVector<int64_t, 4> broadcastDimensions = llvm::to_vector<4>(
          llvm::seq<int64_t>(outputRank - operandType.getRank(), outputRank));
      Value broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(loc,
          broadcastedOutputType, operand, resultExtents,
          rewriter.getI64TensorAttr(broadcastDimensions));
      broadcastedOperands.push_back(broadcast);
    }

    Value mhloOp = rewriter.create<MhloOp<ElementwiseVariadicOp>>(
        loc, op->getResultTypes(), broadcastedOperands);

    rewriter.replaceOp(op, mhloOp);

    return success();
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseVariadicOpLoweringToMhlo<ONNXAddOp>,
      ONNXElementwiseVariadicOpLoweringToMhlo<ONNXSubOp>,
      ONNXElementwiseVariadicOpLoweringToMhlo<ONNXDivOp>,
      ONNXElementwiseUnaryOpLoweringToMhlo<ONNXExpOp>,
      ONNXElementwiseUnaryOpLoweringToMhlo<ONNXReluOp>>(ctx);
}

} // namespace onnx_mlir