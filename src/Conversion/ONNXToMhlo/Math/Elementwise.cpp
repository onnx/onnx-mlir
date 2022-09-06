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
struct MhloDialectOp<ONNXAbsOp> {
  using Op = mhlo::AbsOp;
};

template <>
struct MhloDialectOp<ONNXAndOp> {
  using Op = mhlo::AndOp;
};

template <>
struct MhloDialectOp<ONNXAddOp> {
  using Op = mhlo::AddOp;
};

template <>
struct MhloDialectOp<ONNXCeilOp> {
  using Op = mhlo::CeilOp;
};

template <>
struct MhloDialectOp<ONNXCosOp> {
  using Op = mhlo::CosineOp;
};

template <>
struct MhloDialectOp<ONNXSubOp> {
  using Op = mhlo::SubtractOp;
};

template <>
struct MhloDialectOp<ONNXDivOp> {
  using Op = mhlo::DivOp;
};

template <>
struct MhloDialectOp<ONNXExpOp> {
  using Op = mhlo::ExpOp;
};

template <>
struct MhloDialectOp<ONNXSigmoidOp> {
  using Op = mhlo::LogisticOp;
};

namespace {

template <typename ONNXOp>
void createCompareOp(Value &op, ConversionPatternRewriter &rewriter,
    Location loc, Value &lhs, Value &rhs) {}

template <>
void createCompareOp<ONNXEqualOp>(Value &op,
    ConversionPatternRewriter &rewriter, Location loc, Value &lhs, Value &rhs) {
  op = rewriter.create<mhlo::CompareOp>(
      loc, lhs, rhs, mhlo::ComparisonDirection::EQ);
}

template <>
void createCompareOp<ONNXGreaterOp>(Value &op,
    ConversionPatternRewriter &rewriter, Location loc, Value &lhs, Value &rhs) {
  op = rewriter.create<mhlo::CompareOp>(
      loc, lhs, rhs, mhlo::ComparisonDirection::GT);
}

template <>
void createCompareOp<ONNXGreaterOrEqualOp>(Value &op,
    ConversionPatternRewriter &rewriter, Location loc, Value &lhs, Value &rhs) {
  op = rewriter.create<mhlo::CompareOp>(
      loc, lhs, rhs, mhlo::ComparisonDirection::GE);
}

template <>
void createCompareOp<ONNXLessOp>(Value &op, ConversionPatternRewriter &rewriter,
    Location loc, Value &lhs, Value &rhs) {
  op = rewriter.create<mhlo::CompareOp>(
      loc, lhs, rhs, mhlo::ComparisonDirection::LT);
}

template <>
void createCompareOp<ONNXLessOrEqualOp>(Value &op,
    ConversionPatternRewriter &rewriter, Location loc, Value &lhs, Value &rhs) {
  op = rewriter.create<mhlo::CompareOp>(
      loc, lhs, rhs, mhlo::ComparisonDirection::LE);
}

llvm::SmallVector<Value, 4> getBroadcastedOperands(Operation *op,
    ConversionPatternRewriter &rewriter, Location loc, int64_t outputRank) {
  llvm::SmallVector<Value, 4> broadcastedOperands;
  Type outputType = *op->result_type_begin();
  assert(outputType.isa<ShapedType>() && "output type is not shaped");
  ShapedType outputShapedType = outputType.cast<ShapedType>();
  Type elementType =
      op->getOperands()[0].getType().dyn_cast<ShapedType>().getElementType();
  RankedTensorType broadcastedOutputType =
      RankedTensorType::get(outputShapedType.getShape(), elementType);

  Value resultExtents =
      mlir::hlo::computeNaryElementwiseBroadcastingResultExtents(
          loc, op->getOperands(), rewriter);
  for (Value operand : op->getOperands()) {
    RankedTensorType operandType =
        operand.getType().dyn_cast<RankedTensorType>();
    assert(operandType != nullptr && "operand type is not ranked");
    SmallVector<int64_t, 4> broadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(outputRank - operandType.getRank(), outputRank));
    Value broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(loc,
        broadcastedOutputType, operand, resultExtents,
        rewriter.getI64TensorAttr(broadcastDimensions));
    broadcastedOperands.push_back(broadcast);
  }
  return broadcastedOperands;
}

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
    Value broadcastedZero =
        getShapedZero(loc, rewriter, inpType, inp, resultType);
    Value resultOp =
        rewriter.create<mhlo::MaxOp>(loc, resultType, inp, broadcastedZero);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

template <>
struct ONNXElementwiseUnaryOpLoweringToMhlo<ONNXCastOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(ONNXCastOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(
        StringAttr::get(op->getContext(), ONNXCastOp::getOperationName()),
        op->getLoc());
    ONNXCastOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value inp = adaptor.input();
    Type elementToType = adaptor.to();
    ShapedType inpType = inp.getType().dyn_cast_or_null<ShapedType>();
    if (inpType == nullptr)
      return failure();
    Value resultOp = rewriter.create<mhlo::ConvertOp>(loc, inp, elementToType);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

// Element-wise binary ops lowering to Mhlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseBinaryOp>
struct ONNXElementwiseBinaryOpLoweringToMhlo : public ConversionPattern {
  ONNXElementwiseBinaryOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseBinaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(StringAttr::get(op->getContext(),
                                    ElementwiseBinaryOp::getOperationName()),
        op->getLoc());

    ONNXGenericOpBroadcastedShapeHelper shapeHelper(op);
    DimsExpr empty;
    LogicalResult shapecomputed = shapeHelper.computeShape(operands, empty);
    assert(succeeded(shapecomputed) && "Could not compute output shape");
    int64_t outputRank = shapeHelper.outputRank;
    llvm::SmallVector<Value, 4> broadcastedOperands =
        getBroadcastedOperands(op, rewriter, loc, outputRank);
    Value broadcastedLHS = broadcastedOperands[0];
    Value broadcastedRHS = broadcastedOperands[1];
    Value mhloOp;
    createCompareOp<ElementwiseBinaryOp>(
        mhloOp, rewriter, loc, broadcastedLHS, broadcastedRHS);
    rewriter.replaceOp(op, mhloOp);
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
    llvm::SmallVector<Value, 4> broadcastedOperands =
        getBroadcastedOperands(op, rewriter, loc, outputRank);
    Value mhloOp = rewriter.create<MhloOp<ElementwiseVariadicOp>>(
        loc, op->getResultTypes(), broadcastedOperands);
    rewriter.replaceOp(op, mhloOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseUnaryOpLoweringToMhlo<ONNXAbsOp>,
      ONNXElementwiseVariadicOpLoweringToMhlo<ONNXAddOp>,
      ONNXElementwiseVariadicOpLoweringToMhlo<ONNXAndOp>,
      ONNXElementwiseUnaryOpLoweringToMhlo<ONNXCastOp>,
      ONNXElementwiseUnaryOpLoweringToMhlo<ONNXCeilOp>,
      ONNXElementwiseUnaryOpLoweringToMhlo<ONNXCosOp>,
      ONNXElementwiseVariadicOpLoweringToMhlo<ONNXDivOp>,
      ONNXElementwiseBinaryOpLoweringToMhlo<ONNXEqualOp>,
      ONNXElementwiseBinaryOpLoweringToMhlo<ONNXGreaterOp>,
      ONNXElementwiseBinaryOpLoweringToMhlo<ONNXGreaterOrEqualOp>,
      ONNXElementwiseBinaryOpLoweringToMhlo<ONNXLessOp>,
      ONNXElementwiseBinaryOpLoweringToMhlo<ONNXLessOrEqualOp>,
      ONNXElementwiseVariadicOpLoweringToMhlo<ONNXSubOp>,
      ONNXElementwiseUnaryOpLoweringToMhlo<ONNXExpOp>,
      ONNXElementwiseUnaryOpLoweringToMhlo<ONNXSigmoidOp>,
      ONNXElementwiseUnaryOpLoweringToMhlo<ONNXReluOp>>(ctx);
}

} // namespace onnx_mlir