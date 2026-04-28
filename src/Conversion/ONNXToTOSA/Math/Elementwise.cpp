/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

template <>
struct TOSADialectOp<ONNXNegOp> {
  using Op = mlir::tosa::NegateOp;
};

namespace {

// Element-wise unary ops lowering to TOSA dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
class ONNXElementwiseUnaryOpLoweringToTOSA
    : public OpConversionPattern<ElementwiseUnaryOp> {
public:
  using OpConversionPattern<ElementwiseUnaryOp>::OpConversionPattern;
  using OpAdaptor = typename ElementwiseUnaryOp::Adaptor;
  LogicalResult matchAndRewrite(ElementwiseUnaryOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TOSAOp<ElementwiseUnaryOp>>(
        op, op.getType(), adaptor.getX());
    return success();
  }
};

template <typename ONNXOpT, typename TosaOpT>
class ONNXBinaryElementwiseOpLoweringToTOSA
    : public OpConversionPattern<ONNXOpT> {
public:
  using OpConversionPattern<ONNXOpT>::OpConversionPattern;
  using OpAdaptor = typename ONNXOpT::Adaptor;
  LogicalResult matchAndRewrite(ONNXOpT op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value lhs = adaptor.getA();
    auto lhsType = mlir::dyn_cast<TensorType>(lhs.getType());

    Value rhs = adaptor.getB();
    auto rhsType = mlir::dyn_cast<TensorType>(rhs.getType());

    auto resultType = mlir::dyn_cast<TensorType>(op.getResult().getType());
    if (!lhsType || !rhsType || !resultType) {
      return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
    }

    Type resultElementType = resultType.getElementType();

    if (!resultElementType.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "only int and float are supported");
    }

    if (TosaOpT::template hasTrait<
            mlir::OpTrait::ResultsBroadcastableShape>()) {

      IndexExprBuilderForTosa createTosaIE(rewriter, op->getLoc());
      ONNXBroadcastOpShapeHelper shapeHelper(op, {}, &createTosaIE);
      shapeHelper.computeShapeAndAssertOnFailure();

      if (shapeHelper.hasRankBroadcast()) {
        TosaBuilder tosaBuilder(rewriter, loc);
        llvm::SmallVector<Value, 4> newValues =
            tosaBuilder.equalizeRanks({lhs, rhs});
        lhs = newValues[0];
        rhs = newValues[1];
      }
    }

    rewriter.replaceOpWithNewOp<TosaOpT>(op, op.getType(), lhs, rhs);

    return success();
  }
};

class ONNXSinOpLoweringToTOSA : public OpConversionPattern<ONNXSinOp> {
public:
  using OpConversionPattern<ONNXSinOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXSinOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXSinOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::tosa::SinOp>(
        op, op.getType(), adaptor.getInput());
    return success();
  }
};

class ONNXCosOpLoweringToTOSA : public OpConversionPattern<ONNXCosOp> {
public:
  using OpConversionPattern<ONNXCosOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXCosOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXCosOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::tosa::CosOp>(
        op, op.getType(), adaptor.getInput());
    return success();
  }
};

class ONNXFloorOpLoweringToTOSA : public OpConversionPattern<ONNXFloorOp> {
public:
  using OpConversionPattern<ONNXFloorOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXFloorOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXFloorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto scalarType = getElementTypeOrSelf(adaptor.getX());
    if (!isTOSAFloat(scalarType))
      return rewriter.notifyMatchFailure(
          op, "`tosa.floor` only supports float types");

    rewriter.replaceOpWithNewOp<mlir::tosa::FloorOp>(
        op, op.getType(), adaptor.getX());
    return success();
  }
};

class ONNXReluOpLoweringToTOSA : public OpConversionPattern<ONNXReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getX();

    // Quantized types are not supported right now (in type conversion).
    // Once they are, the input should be rescaled for quantized types. (TBD)
    // Maps to `tosa.clamp` which has both int and fp limits.
    auto inputElementType =
        llvm::cast<TensorType>(op.getType()).getElementType();
    if (llvm::isa<IntegerType>(inputElementType)) {
      auto minClamp = rewriter.getI64IntegerAttr(0);
      auto maxClamp =
          rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max());
      rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(
          op, op.getType(), input, minClamp, maxClamp);
    } else {
      auto minClamp = rewriter.getF32FloatAttr(0.0f);
      auto maxClamp =
          rewriter.getF32FloatAttr(std::numeric_limits<float>::max());
      rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(
          op, op.getType(), input, minClamp, maxClamp);
    }
    return success();
  }
};

// Extract a scalar ElementsAttr from a value defined either by
// onnx.Constant (before conversion) or tosa.const (after conversion).
static ElementsAttr getScalarConstantElementsAttr(Value v) {
  if (auto onnxConst =
          mlir::dyn_cast_or_null<ONNXConstantOp>(v.getDefiningOp()))
    return mlir::dyn_cast_or_null<ElementsAttr>(onnxConst.getValueAttr());
  if (auto tosaConst =
          mlir::dyn_cast_or_null<mlir::tosa::ConstOp>(v.getDefiningOp()))
    return tosaConst.getValues();
  return nullptr;
}

class ONNXClipOpLoweringToTOSA : public OpConversionPattern<ONNXClipOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXClipOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    Value min = adaptor.getMin();
    Value max = adaptor.getMax();

    auto inputType = mlir::dyn_cast<TensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
    Type elementType = inputType.getElementType();
    if (!elementType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "only int and float types are supported");

    Attribute minAttr;
    Attribute maxAttr;

    if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
      const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
      APFloat minVal = APFloat::getLargest(semantics, /*Negative=*/true);
      APFloat maxVal = APFloat::getLargest(semantics, /*Negative=*/false);

      if (!isNoneValue(min)) {
        ElementsAttr minElems = getScalarConstantElementsAttr(min);
        if (!minElems)
          return rewriter.notifyMatchFailure(
              op, "min must be a constant for tosa.clamp");
        minVal = *minElems.getValues<APFloat>().begin();
      }
      if (!isNoneValue(max)) {
        ElementsAttr maxElems = getScalarConstantElementsAttr(max);
        if (!maxElems)
          return rewriter.notifyMatchFailure(
              op, "max must be a constant for tosa.clamp");
        maxVal = *maxElems.getValues<APFloat>().begin();
      }
      minAttr = rewriter.getFloatAttr(elementType, minVal);
      maxAttr = rewriter.getFloatAttr(elementType, maxVal);
    } else {
      auto intType = mlir::cast<IntegerType>(elementType);
      unsigned width = intType.getWidth();
      APInt minVal = intType.isUnsigned() ? APInt::getMinValue(width)
                                          : APInt::getSignedMinValue(width);
      APInt maxVal = intType.isUnsigned() ? APInt::getMaxValue(width)
                                          : APInt::getSignedMaxValue(width);

      if (!isNoneValue(min)) {
        ElementsAttr minElems = getScalarConstantElementsAttr(min);
        if (!minElems)
          return rewriter.notifyMatchFailure(
              op, "min must be a constant for tosa.clamp");
        minVal = *minElems.getValues<APInt>().begin();
      }
      if (!isNoneValue(max)) {
        ElementsAttr maxElems = getScalarConstantElementsAttr(max);
        if (!maxElems)
          return rewriter.notifyMatchFailure(
              op, "max must be a constant for tosa.clamp");
        maxVal = *maxElems.getValues<APInt>().begin();
      }
      minAttr = rewriter.getIntegerAttr(elementType, minVal);
      maxAttr = rewriter.getIntegerAttr(elementType, maxVal);
    }

    rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(
        op, op.getType(), input, minAttr, maxAttr);
    return success();
  }
};

class ONNXDivOpLoweringToTOSA : public OpConversionPattern<ONNXDivOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXDivOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();
    auto resultType = mlir::cast<TensorType>(op.getResult().getType());
    Type resultElementType = resultType.getElementType();

    TosaBuilder tosaBuilder(rewriter, op->getLoc());

    if (resultElementType.isSignlessInteger(32)) {
      // tosa::IntDivOp takes 32-but signless integers as inputs
      Value divOp = tosaBuilder.intdiv(lhs, rhs);
      rewriter.replaceOp(op, {divOp});
      return success();
    }
    // If it is not a 32-bit signless integer, decompose ONNXDivOp into
    // tosa::ReciprocalOp and tosa::MulOp
    Value reciprocalOp = tosaBuilder.reciprocal(rhs);
    Value mulOp = tosaBuilder.mul(lhs, reciprocalOp);
    rewriter.replaceOp(op, {mulOp});
    return success();
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseUnaryOpLoweringToTOSA<ONNXNegOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXAddOp, mlir::tosa::AddOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXSubOp, mlir::tosa::SubOp>,
      ONNXSinOpLoweringToTOSA, ONNXCosOpLoweringToTOSA,
      ONNXFloorOpLoweringToTOSA, ONNXReluOpLoweringToTOSA,
      ONNXClipOpLoweringToTOSA, ONNXDivOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
