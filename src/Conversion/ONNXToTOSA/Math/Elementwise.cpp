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

// Create a splatted constant tensor whose element type matches `elementType`
// and whose shape is the same rank as `referenceShape` with all dims equal
// to 1 so it can broadcast against tensors of that rank.
static Value createFloatSplatConst(PatternRewriter &rewriter, Location loc,
    double value, FloatType elementType, ArrayRef<int64_t> referenceShape) {
  APFloat apVal(value);
  bool losesInfo = false;
  apVal.convert(elementType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
      &losesInfo);
  auto constType = tosa::reduceAxisToOne(referenceShape, elementType);
  auto constAttr = DenseElementsAttr::get(constType, apVal);
  return mlir::tosa::ConstOp::create(rewriter, loc, constType, constAttr);
}

class ONNXGeluOpLoweringToTOSA : public OpConversionPattern<ONNXGeluOp> {
public:
  using OpConversionPattern<ONNXGeluOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXGeluOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXGeluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value x = adaptor.getX();

    auto inputType = mlir::dyn_cast<RankedTensorType>(x.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");
    auto elementType = mlir::dyn_cast<FloatType>(inputType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "tosa.gelu lowering only supports float types");

    TosaBuilder tosaBuilder(rewriter, loc);
    StringRef approximate = adaptor.getApproximate();
    ArrayRef<int64_t> shape = inputType.getShape();
    Value half = createFloatSplatConst(rewriter, loc, 0.5, elementType, shape);
    Value one = createFloatSplatConst(rewriter, loc, 1.0, elementType, shape);

    Value inner;
    if (approximate == "none") {
      // y = 0.5 * x * (1 + erf(x / sqrt(2)))
      Value invSqrt2 = createFloatSplatConst(
          rewriter, loc, 0.70710678118654752440, elementType, shape);
      Value scaled = tosaBuilder.mul(x, invSqrt2);
      inner =
          mlir::tosa::ErfOp::create(rewriter, loc, scaled.getType(), scaled);
    } else if (approximate == "tanh") {
      // y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      Value coeff =
          createFloatSplatConst(rewriter, loc, 0.044715, elementType, shape);
      Value sqrt2OverPi = createFloatSplatConst(
          rewriter, loc, 0.79788456080286535588, elementType, shape);
      Value xSquared = tosaBuilder.mul(x, x);
      Value xCubed = tosaBuilder.mul(xSquared, x);
      Value coeffXCubed = tosaBuilder.mul(coeff, xCubed);
      Value sum = tosaBuilder.binaryOp<mlir::tosa::AddOp>(x, coeffXCubed);
      Value scaled = tosaBuilder.mul(sqrt2OverPi, sum);
      inner =
          mlir::tosa::TanhOp::create(rewriter, loc, scaled.getType(), scaled);
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported 'approximate' attribute value");
    }

    Value addOne = tosaBuilder.binaryOp<mlir::tosa::AddOp>(inner, one);
    Value mulX = tosaBuilder.mul(x, addOne);
    Value result = tosaBuilder.mul(mulX, half);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ONNXErfOpLoweringToTOSA : public OpConversionPattern<ONNXErfOp> {
public:
  using OpConversionPattern<ONNXErfOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXErfOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXErfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::tosa::ErfOp>(
        op, op.getType(), adaptor.getInput());
    return success();
  }
};

class ONNXTanhOpLoweringToTOSA : public OpConversionPattern<ONNXTanhOp> {
public:
  using OpConversionPattern<ONNXTanhOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXTanhOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXTanhOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::tosa::TanhOp>(
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
      ONNXSinOpLoweringToTOSA, ONNXCosOpLoweringToTOSA, ONNXErfOpLoweringToTOSA,
      ONNXTanhOpLoweringToTOSA, ONNXGeluOpLoweringToTOSA,
      ONNXFloorOpLoweringToTOSA, ONNXReluOpLoweringToTOSA,
      ONNXClipOpLoweringToTOSA, ONNXDivOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
