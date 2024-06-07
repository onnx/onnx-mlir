/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "stablehlo/dialect/BroadcastUtils.h"

using namespace mlir;

namespace onnx_mlir {

template <>
struct StablehloDialectOp<ONNXAbsOp> {
  using Op = stablehlo::AbsOp;
};

template <>
struct StablehloDialectOp<ONNXAndOp> {
  using Op = stablehlo::AndOp;
};

template <>
struct StablehloDialectOp<ONNXAddOp> {
  using Op = stablehlo::AddOp;
};

template <>
struct StablehloDialectOp<ONNXCeilOp> {
  using Op = stablehlo::CeilOp;
};

template <>
struct StablehloDialectOp<ONNXCosOp> {
  using Op = stablehlo::CosineOp;
};

template <>
struct StablehloDialectOp<ONNXDivOp> {
  using Op = stablehlo::DivOp;
};

template <>
struct StablehloDialectOp<ONNXExpOp> {
  using Op = stablehlo::ExpOp;
};

template <>
struct StablehloDialectOp<ONNXLogOp> {
  using Op = stablehlo::LogOp;
};

template <>
struct StablehloDialectOp<ONNXMaxOp> {
  using Op = stablehlo::MaxOp;
};

template <>
struct StablehloDialectOp<ONNXMinOp> {
  using Op = stablehlo::MinOp;
};

template <>
struct StablehloDialectOp<ONNXMulOp> {
  using Op = stablehlo::MulOp;
};

template <>
struct StablehloDialectOp<ONNXNegOp> {
  using Op = stablehlo::NegOp;
};

template <>
struct StablehloDialectOp<ONNXPowOp> {
  using Op = stablehlo::PowOp;
};

template <>
struct StablehloDialectOp<ONNXSigmoidOp> {
  using Op = stablehlo::LogisticOp;
};

template <>
struct StablehloDialectOp<ONNXSinOp> {
  using Op = stablehlo::SineOp;
};

template <>
struct StablehloDialectOp<ONNXSqrtOp> {
  using Op = stablehlo::SqrtOp;
};

template <>
struct StablehloDialectOp<ONNXSubOp> {
  using Op = stablehlo::SubtractOp;
};

template <>
struct StablehloDialectOp<ONNXTanhOp> {
  using Op = stablehlo::TanhOp;
};

template <>
struct StablehloDialectOp<ONNXWhereOp> {
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

// Element-wise unary ops lowering to Stablehlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLoweringToStablehlo : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseUnaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value stableHloOp = rewriter.create<StablehloOp<ElementwiseUnaryOp>>(
        loc, op->getResultTypes(), op->getOperands());
    rewriter.replaceOp(op, stableHloOp);
    return success();
  }
};

// ONNXElu(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0.
template <>
struct ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXEluOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ONNXEluOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXEluOpAdaptor operandAdaptor(operands);
    ONNXEluOp EluOp = llvm::cast<ONNXEluOp>(op);
    double alpha = EluOp.getAlpha().convertToDouble();

    Type resultType = *op->result_type_begin();
    Value inp = operandAdaptor.getX();
    ShapedType inpType = mlir::dyn_cast_or_null<ShapedType>(inp.getType());
    if (inpType == nullptr)
      return failure();
    Value alphaVal = getShapedFloat(loc, rewriter, alpha, inp);
    Value oneVal = getShapedFloat(loc, rewriter, 1.0f, inp);
    Value expVal = rewriter.create<stablehlo::ExpOp>(loc, inp);
    Value subVal = rewriter.create<stablehlo::SubtractOp>(loc, expVal, oneVal);
    Value mulVal = rewriter.create<stablehlo::MulOp>(loc, alphaVal, subVal);
    Value broadcastedZero = getShapedZero(loc, rewriter, inp);
    Value compareGeZero = rewriter.create<stablehlo::CompareOp>(
        loc, inp, broadcastedZero, stablehlo::ComparisonDirection::GE);
    Value resultOp = rewriter.create<stablehlo::SelectOp>(
        loc, resultType, compareGeZero, inp, mulVal);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

// ONNXHardSigmoid(x) = max(0, min(1, alpha * x + beta))
template <>
struct ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXHardSigmoidOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ONNXHardSigmoidOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXHardSigmoidOpAdaptor operandAdaptor(operands);
    ONNXHardSigmoidOp HardSigmoidOp = llvm::cast<ONNXHardSigmoidOp>(op);
    double alpha = HardSigmoidOp.getAlpha().convertToDouble();
    double beta = HardSigmoidOp.getBeta().convertToDouble();
    Value inp = operandAdaptor.getX();
    ShapedType inpType = mlir::dyn_cast_or_null<ShapedType>(inp.getType());
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

// ONNXReluOp(x) is implemented using Stablehlo Max(x, 0)
template <>
struct ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXReluOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ONNXReluOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXReluOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value inp = adaptor.getX();
    ShapedType inpType = mlir::dyn_cast_or_null<ShapedType>(inp.getType());
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
struct ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXLeakyReluOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ONNXLeakyReluOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXLeakyReluOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value inp = adaptor.getX();
    llvm::APFloat alpha = adaptor.getAlpha();
    ShapedType inpType = mlir::dyn_cast_or_null<ShapedType>(inp.getType());
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
struct ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXCastOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ONNXCastOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXCastOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value inp = adaptor.getInput();
    Type elementToType = adaptor.getTo();
    ShapedType inpType = mlir::dyn_cast_or_null<ShapedType>(inp.getType());
    if (inpType == nullptr)
      return failure();
    Value resultOp =
        rewriter.create<stablehlo::ConvertOp>(loc, inp, elementToType);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

// Element-wise compare binary ops lowering to Stablehlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseBinaryOp>
struct ONNXElementwiseCompareBinaryOpLoweringToStablehlo
    : public ConversionPattern {
  ONNXElementwiseCompareBinaryOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseBinaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    // Prior code here used the "analysis" version that did not generate code.
    // Since code is actually not needed here at this time, one could use
    // IndexExprBuilderForAnalysis createIE(loc) instead.
    IndexExprBuilderForStablehlo createShapeIE(rewriter, loc);
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

// Element-wise compare binary ops lowering to Stablehlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseBinaryOp>
struct ONNXElementwiseBinaryOpLoweringToStablehlo : public ConversionPattern {
  ONNXElementwiseBinaryOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseBinaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    // Prior code here used the "analysis" version that did not generate code.
    // Since code is actually not needed here at this time, one could use
    // IndexExprBuilderForAnalysis createIE(loc) instead.
    IndexExprBuilderForStablehlo createShapeIE(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &createShapeIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    int64_t outputRank = shapeHelper.outputRank;
    llvm::SmallVector<Value, 4> broadcastedOperands =
        getBroadcastedOperands(op, rewriter, loc, outputRank);
    Value broadcastedLHS = broadcastedOperands[0];
    Value broadcastedRHS = broadcastedOperands[1];
    Value stableHloOp = rewriter.create<StablehloOp<ElementwiseBinaryOp>>(
        loc, *op->result_type_begin(), broadcastedLHS, broadcastedRHS);
    rewriter.replaceOp(op, stableHloOp);
    return success();
  }
};

// ONNXPReluOp(x) = alpha * x if x < 0 else x.
template <>
struct ONNXElementwiseBinaryOpLoweringToStablehlo<ONNXPReluOp>
    : public ConversionPattern {
  ONNXElementwiseBinaryOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ONNXPReluOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    // Prior code here used the "analysis" version that did not generate code.
    // Since code is actually not needed here at this time, one could use
    // IndexExprBuilderForAnalysis createIE(loc) instead.
    IndexExprBuilderForStablehlo createShapeIE(rewriter, loc);
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

// Element-wise variadic ops lowering to Stablehlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLoweringToStablehlo : public ConversionPattern {
  ONNXElementwiseVariadicOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseVariadicOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    // Prior code here used the "analysis" version that did not generate code.
    // Since code is actually not needed here at this time, one could use
    // IndexExprBuilderForAnalysis createIE(loc) instead.
    IndexExprBuilderForStablehlo createShapeIE(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &createShapeIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    int64_t outputRank = shapeHelper.outputRank;
    llvm::SmallVector<Value, 4> broadcastedOperands =
        getBroadcastedOperands(op, rewriter, loc, outputRank);
    Value stableHloOp = rewriter.create<StablehloOp<ElementwiseVariadicOp>>(
        loc, op->getResultTypes(), broadcastedOperands);
    rewriter.replaceOp(op, stableHloOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXAbsOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXCastOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXCeilOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXCosOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXEluOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXExpOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXHardSigmoidOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXLeakyReluOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXLogOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXNegOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXSigmoidOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXSinOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXSqrtOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXReluOp>,
      ONNXElementwiseUnaryOpLoweringToStablehlo<ONNXTanhOp>,
      ONNXElementwiseCompareBinaryOpLoweringToStablehlo<ONNXEqualOp>,
      ONNXElementwiseCompareBinaryOpLoweringToStablehlo<ONNXGreaterOp>,
      ONNXElementwiseCompareBinaryOpLoweringToStablehlo<ONNXGreaterOrEqualOp>,
      ONNXElementwiseCompareBinaryOpLoweringToStablehlo<ONNXLessOp>,
      ONNXElementwiseCompareBinaryOpLoweringToStablehlo<ONNXLessOrEqualOp>,
      ONNXElementwiseBinaryOpLoweringToStablehlo<ONNXPowOp>,
      ONNXElementwiseBinaryOpLoweringToStablehlo<ONNXPReluOp>,
      ONNXElementwiseVariadicOpLoweringToStablehlo<ONNXAddOp>,
      ONNXElementwiseVariadicOpLoweringToStablehlo<ONNXAndOp>,
      ONNXElementwiseVariadicOpLoweringToStablehlo<ONNXDivOp>,
      ONNXElementwiseVariadicOpLoweringToStablehlo<ONNXMaxOp>,
      ONNXElementwiseVariadicOpLoweringToStablehlo<ONNXMinOp>,
      ONNXElementwiseVariadicOpLoweringToStablehlo<ONNXMulOp>,
      ONNXElementwiseVariadicOpLoweringToStablehlo<ONNXSubOp>,
      ONNXElementwiseVariadicOpLoweringToStablehlo<ONNXWhereOp>>(ctx);
}

} // namespace onnx_mlir
