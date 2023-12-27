/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStableHlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStableHlo/ONNXToStableHloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "stablehlo/dialect/BroadcastUtils.h"

using namespace mlir;

namespace onnx_mlir {

template <>
struct StableHloDialectOp<ONNXAbsOp> {
  using Op = stablehlo::AbsOp;
};

template <>
struct StableHloDialectOp<ONNXAndOp> {
  using Op = stablehlo::AndOp;
};

template <>
struct StableHloDialectOp<ONNXAddOp> {
  using Op = stablehlo::AddOp;
};

template <>
struct StableHloDialectOp<ONNXCeilOp> {
  using Op = stablehlo::CeilOp;
};

template <>
struct StableHloDialectOp<ONNXCosOp> {
  using Op = stablehlo::CosineOp;
};

template <>
struct StableHloDialectOp<ONNXDivOp> {
  using Op = stablehlo::DivOp;
};

template <>
struct StableHloDialectOp<ONNXExpOp> {
  using Op = stablehlo::ExpOp;
};

template <>
struct StableHloDialectOp<ONNXLogOp> {
  using Op = stablehlo::LogOp;
};

template <>
struct StableHloDialectOp<ONNXMaxOp> {
  using Op = stablehlo::MaxOp;
};

template <>
struct StableHloDialectOp<ONNXMinOp> {
  using Op = stablehlo::MinOp;
};

template <>
struct StableHloDialectOp<ONNXMulOp> {
  using Op = stablehlo::MulOp;
};

template <>
struct StableHloDialectOp<ONNXNegOp> {
  using Op = stablehlo::NegOp;
};

template <>
struct StableHloDialectOp<ONNXPowOp> {
  using Op = stablehlo::PowOp;
};

template <>
struct StableHloDialectOp<ONNXSigmoidOp> {
  using Op = stablehlo::LogisticOp;
};

template <>
struct StableHloDialectOp<ONNXSinOp> {
  using Op = stablehlo::SineOp;
};

template <>
struct StableHloDialectOp<ONNXSqrtOp> {
  using Op = stablehlo::SqrtOp;
};

template <>
struct StableHloDialectOp<ONNXSubOp> {
  using Op = stablehlo::SubtractOp;
};

template <>
struct StableHloDialectOp<ONNXTanhOp> {
  using Op = stablehlo::TanhOp;
};

template <>
struct StableHloDialectOp<ONNXWhereOp> {
  using Op = stablehlo::SelectOp;
};

namespace {

template <typename ONNXOp>
void createCompareOp(Value &op, ConversionPatternRewriter &rewriter,
    Location loc, Value &lhs, Value &rhs) {}

template <>
void createCompareOp<ONNXEqualOp>(Value &op,
    ConversionPatternRewriter &rewriter, Location loc, Value &lhs, Value &rhs) {
  op = rewriter.create<stablehlo::CompareOp>(
      loc, lhs, rhs, stablehlo::ComparisonDirection::EQ);
}

template <>
void createCompareOp<ONNXGreaterOp>(Value &op,
    ConversionPatternRewriter &rewriter, Location loc, Value &lhs, Value &rhs) {
  op = rewriter.create<stablehlo::CompareOp>(
      loc, lhs, rhs, stablehlo::ComparisonDirection::GT);
}

template <>
void createCompareOp<ONNXGreaterOrEqualOp>(Value &op,
    ConversionPatternRewriter &rewriter, Location loc, Value &lhs, Value &rhs) {
  op = rewriter.create<stablehlo::CompareOp>(
      loc, lhs, rhs, stablehlo::ComparisonDirection::GE);
}

template <>
void createCompareOp<ONNXLessOp>(Value &op, ConversionPatternRewriter &rewriter,
    Location loc, Value &lhs, Value &rhs) {
  op = rewriter.create<stablehlo::CompareOp>(
      loc, lhs, rhs, stablehlo::ComparisonDirection::LT);
}

template <>
void createCompareOp<ONNXLessOrEqualOp>(Value &op,
    ConversionPatternRewriter &rewriter, Location loc, Value &lhs, Value &rhs) {
  op = rewriter.create<stablehlo::CompareOp>(
      loc, lhs, rhs, stablehlo::ComparisonDirection::LE);
}

// Element-wise unary ops lowering to StableHlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLoweringToStableHlo : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseUnaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value stableHloOp = rewriter.create<StableHloOp<ElementwiseUnaryOp>>(
        loc, op->getResultTypes(), op->getOperands());
    rewriter.replaceOp(op, stableHloOp);
    return success();
  }
};

// ONNXHardSigmoid(x) = max(0, min(1, alpha * x + beta))
template <>
struct ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXHardSigmoidOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(ONNXHardSigmoidOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXHardSigmoidOpAdaptor operandAdaptor(operands);
    ONNXHardSigmoidOp HardSigmoidOp = llvm::cast<ONNXHardSigmoidOp>(op);
    double alpha = HardSigmoidOp.getAlpha().convertToDouble();
    double beta = HardSigmoidOp.getBeta().convertToDouble();
    Value inp = operandAdaptor.getX();
    ShapedType inpType = inp.getType().dyn_cast_or_null<ShapedType>();
    if (inpType == nullptr)
      return failure();
    Value alphaVal = getShapedFloat(loc, rewriter, alpha, inp);
    Value betaVal = getShapedFloat(loc, rewriter, beta, inp);
    Value zeroVal = getShapedFloat(loc, rewriter, 0.0f, inp);
    Value oneVal = getShapedFloat(loc, rewriter, 1.0f, inp);
    Value productVal = rewriter.create<stablehlo::MulOp>(loc, inp, alphaVal);
    Value sumVal = rewriter.create<stablehlo::AddOp>(loc, productVal, betaVal);
    Value resultOp =
        rewriter.create<stablehlo::ClampOp>(loc, zeroVal, sumVal, oneVal);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

// ONNXReluOp(x) is implemented using StableHlo Max(x, 0)
template <>
struct ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXReluOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(ONNXReluOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXReluOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value inp = adaptor.getX();
    ShapedType inpType = inp.getType().dyn_cast_or_null<ShapedType>();
    if (inpType == nullptr)
      return failure();
    Type resultType = *op->result_type_begin();
    Value broadcastedZero = getShapedZero(loc, rewriter, inp);
    Value resultOp = rewriter.create<stablehlo::MaxOp>(
        loc, resultType, inp, broadcastedZero);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

// ONNXLeakyReluOp(x) = alpha * x if x < 0 else x.
template <>
struct ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXLeakyReluOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(ONNXLeakyReluOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXLeakyReluOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value inp = adaptor.getX();
    llvm::APFloat alpha = adaptor.getAlpha();
    ShapedType inpType = inp.getType().dyn_cast_or_null<ShapedType>();
    if (inpType == nullptr)
      return failure();
    Type resultType = *op->result_type_begin();
    Value alphaVal = getShapedFloat(loc, rewriter, alpha, inp);
    Value leakyActivationVal =
        rewriter.create<stablehlo::MulOp>(loc, inp, alphaVal);
    Value broadcastedZero = getShapedZero(loc, rewriter, inp);
    Value compareGtZero = rewriter.create<stablehlo::CompareOp>(
        loc, inp, broadcastedZero, stablehlo::ComparisonDirection::GT);
    Value resultOp = rewriter.create<stablehlo::SelectOp>(
        loc, resultType, compareGtZero, inp, leakyActivationVal);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

template <>
struct ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXCastOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(ONNXCastOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXCastOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value inp = adaptor.getInput();
    Type elementToType = adaptor.getTo();
    ShapedType inpType = inp.getType().dyn_cast_or_null<ShapedType>();
    if (inpType == nullptr)
      return failure();
    Value resultOp =
        rewriter.create<stablehlo::ConvertOp>(loc, inp, elementToType);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

// Element-wise compare binary ops lowering to StableHlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseBinaryOp>
struct ONNXElementwiseCompareBinaryOpLoweringToStableHlo
    : public ConversionPattern {
  ONNXElementwiseCompareBinaryOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseBinaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    // Prior code here used the "analysis" version that did not generate code.
    // Since code is actually not needed here at this time, one could use
    // IndexExprBuilderForAnalysis createIE(loc) instead.
    IndexExprBuilderForStableHlo createShapeIE(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &createShapeIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    int64_t outputRank = shapeHelper.outputRank;
    llvm::SmallVector<Value, 4> broadcastedOperands =
        getBroadcastedOperands(op, rewriter, loc, outputRank);
    Value broadcastedLHS = broadcastedOperands[0];
    Value broadcastedRHS = broadcastedOperands[1];
    Value stableHloOp;
    createCompareOp<ElementwiseBinaryOp>(
        stableHloOp, rewriter, loc, broadcastedLHS, broadcastedRHS);
    rewriter.replaceOp(op, stableHloOp);
    return success();
  }
};

// Element-wise compare binary ops lowering to StableHlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseBinaryOp>
struct ONNXElementwiseBinaryOpLoweringToStableHlo : public ConversionPattern {
  ONNXElementwiseBinaryOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseBinaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    // Prior code here used the "analysis" version that did not generate code.
    // Since code is actually not needed here at this time, one could use
    // IndexExprBuilderForAnalysis createIE(loc) instead.
    IndexExprBuilderForStableHlo createShapeIE(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &createShapeIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    int64_t outputRank = shapeHelper.outputRank;
    llvm::SmallVector<Value, 4> broadcastedOperands =
        getBroadcastedOperands(op, rewriter, loc, outputRank);
    Value broadcastedLHS = broadcastedOperands[0];
    Value broadcastedRHS = broadcastedOperands[1];
    Value stableHloOp = rewriter.create<StableHloOp<ElementwiseBinaryOp>>(
        loc, *op->result_type_begin(), broadcastedLHS, broadcastedRHS);
    rewriter.replaceOp(op, stableHloOp);
    return success();
  }
};

// ONNXPReluOp(x) = alpha * x if x < 0 else x.
template <>
struct ONNXElementwiseBinaryOpLoweringToStableHlo<ONNXPReluOp>
    : public ConversionPattern {
  ONNXElementwiseBinaryOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(ONNXPReluOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    // Prior code here used the "analysis" version that did not generate code.
    // Since code is actually not needed here at this time, one could use
    // IndexExprBuilderForAnalysis createIE(loc) instead.
    IndexExprBuilderForStableHlo createShapeIE(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &createShapeIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    int64_t outputRank = shapeHelper.outputRank;
    llvm::SmallVector<Value, 4> broadcastedOperands =
        getBroadcastedOperands(op, rewriter, loc, outputRank);
    Value inp = broadcastedOperands[0];
    Value broadcastedSlope = broadcastedOperands[1];
    Type resultType = *op->result_type_begin();
    Value PReluActivationVal =
        rewriter.create<stablehlo::MulOp>(loc, inp, broadcastedSlope);
    Value broadcastedZero = getShapedZero(loc, rewriter, inp);
    Value compareGtZero = rewriter.create<stablehlo::CompareOp>(
        loc, inp, broadcastedZero, stablehlo::ComparisonDirection::GT);
    Value resultOp = rewriter.create<stablehlo::SelectOp>(
        loc, resultType, compareGtZero, inp, PReluActivationVal);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

// Element-wise variadic ops lowering to StableHlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLoweringToStableHlo : public ConversionPattern {
  ONNXElementwiseVariadicOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseVariadicOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    // Prior code here used the "analysis" version that did not generate code.
    // Since code is actually not needed here at this time, one could use
    // IndexExprBuilderForAnalysis createIE(loc) instead.
    IndexExprBuilderForStableHlo createShapeIE(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &createShapeIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    int64_t outputRank = shapeHelper.outputRank;
    llvm::SmallVector<Value, 4> broadcastedOperands =
        getBroadcastedOperands(op, rewriter, loc, outputRank);
    Value stableHloOp = rewriter.create<StableHloOp<ElementwiseVariadicOp>>(
        loc, op->getResultTypes(), broadcastedOperands);
    rewriter.replaceOp(op, stableHloOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToStableHloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXAbsOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXCastOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXCeilOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXCosOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXExpOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXHardSigmoidOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXLeakyReluOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXLogOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXNegOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXSigmoidOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXSinOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXSqrtOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXReluOp>,
      ONNXElementwiseUnaryOpLoweringToStableHlo<ONNXTanhOp>,
      ONNXElementwiseCompareBinaryOpLoweringToStableHlo<ONNXEqualOp>,
      ONNXElementwiseCompareBinaryOpLoweringToStableHlo<ONNXGreaterOp>,
      ONNXElementwiseCompareBinaryOpLoweringToStableHlo<ONNXGreaterOrEqualOp>,
      ONNXElementwiseCompareBinaryOpLoweringToStableHlo<ONNXLessOp>,
      ONNXElementwiseCompareBinaryOpLoweringToStableHlo<ONNXLessOrEqualOp>,
      ONNXElementwiseBinaryOpLoweringToStableHlo<ONNXPowOp>,
      ONNXElementwiseBinaryOpLoweringToStableHlo<ONNXPReluOp>,
      ONNXElementwiseVariadicOpLoweringToStableHlo<ONNXAddOp>,
      ONNXElementwiseVariadicOpLoweringToStableHlo<ONNXAndOp>,
      ONNXElementwiseVariadicOpLoweringToStableHlo<ONNXDivOp>,
      ONNXElementwiseVariadicOpLoweringToStableHlo<ONNXMaxOp>,
      ONNXElementwiseVariadicOpLoweringToStableHlo<ONNXMinOp>,
      ONNXElementwiseVariadicOpLoweringToStableHlo<ONNXMulOp>,
      ONNXElementwiseVariadicOpLoweringToStableHlo<ONNXSubOp>,
      ONNXElementwiseVariadicOpLoweringToStableHlo<ONNXWhereOp>>(ctx);
}

} // namespace onnx_mlir
