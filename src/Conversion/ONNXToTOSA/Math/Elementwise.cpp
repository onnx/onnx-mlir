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

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include <src/Conversion/ONNXToTOSA/DialectBuilder.hpp>

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

    Value lhs = adaptor.getA();
    auto lhsType = lhs.getType().dyn_cast<TensorType>();

    Value rhs = adaptor.getB();
    auto rhsType = rhs.getType().dyn_cast<TensorType>();

    auto resultType = op.getResult().getType().template dyn_cast<TensorType>();
    if (!lhsType || !rhsType || !resultType) {
      return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
    }

    Type resultElementType = resultType.getElementType();

    if (!resultElementType.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "only int and float are supported");
    }

    rewriter.replaceOpWithNewOp<TosaOpT>(op, op.getType(), lhs, rhs);

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
    }

    rewriter.replaceOpWithNewOp<tosa::FloorOp>(
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
    rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(op, op.getType(), input,
        rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max()),
        rewriter.getF32FloatAttr(0.0f),
        rewriter.getF32FloatAttr(std::numeric_limits<float>::max()));
    return success();
  }
};

// Support for prelu/leakyrelu adapted from tensorflow to tosa implementation
static LogicalResult LegalizeFloatingPointPrelu(Operation *op,
    PatternRewriter &rewriter, Value input, float alpha,
    TensorType outputType) {
  auto loc = op->getLoc();
  TosaBuilder tosaBuilder(rewriter, loc);
  Value constZero = tosaBuilder.getConst(0.0, outputType.getShape());

  auto mul = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(rewriter, op->getLoc(),
      outputType, input, tosaBuilder.getConst(alpha, outputType.getShape()),
      /*shift=*/0);

  auto greaterEqual =
      tosa::CreateOpAndInfer<mlir::tosa::GreaterEqualOp>(rewriter, op->getLoc(),
          UnrankedTensorType::get(rewriter.getI1Type()), input, constZero);

  tosa::CreateReplaceOpAndInfer<mlir::tosa::SelectOp>(
      rewriter, op, outputType, greaterEqual, input, mul.getResult());

  return success();
}

class ONNXLeakyReluOpLoweringToTOSA
    : public OpConversionPattern<ONNXLeakyReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = ONNXLeakyReluOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXLeakyReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto outputType = op.getResult().getType().cast<TensorType>();

    if (!outputType.getElementType().isF32()) {
      return rewriter.notifyMatchFailure(op, "only float is supported");
    }

    // ONNX docs: alpha : float (default 0.01)
    float alpha = 0.01;
    FloatAttr alphaAttr = adaptor.alphaAttr();
    if (alphaAttr) {
      // No easy interface in MLIR to get value as float
      alpha = alphaAttr.getValueAsDouble();
    }
    return LegalizeFloatingPointPrelu(
        op, rewriter, adaptor.X(), alpha, outputType);
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseUnaryOpLoweringToTOSA<ONNXNegOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXAddOp, mlir::tosa::AddOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXSubOp, mlir::tosa::SubOp>,
      ONNXFloorOpLoweringToTOSA, ONNXReluOpLoweringToTOSA,
      ONNXLeakyReluOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
