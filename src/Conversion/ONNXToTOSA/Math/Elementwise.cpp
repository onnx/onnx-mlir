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

namespace {

struct IsFloat {
  static LogicalResult checkType(
      ConversionPatternRewriter &rewriter, Type scalarType, Operation *op) {
    if (!isTOSAFloat(scalarType)) {
      return rewriter.notifyMatchFailure(
          op, "this operation only support float types");
    }
    return success();
  }
};

struct IsIntOrFloat {
  static LogicalResult checkType(
      ConversionPatternRewriter &rewriter, Type scalarType, Operation *op) {
    if (!isTOSAFloat(scalarType) && !isTOSASignedInt(scalarType)) {
      return rewriter.notifyMatchFailure(
          op, "this operation only support signed integer or float types");
    }
    return success();
  }
};

template <typename OpAdaptorT>
LogicalResult checkBasicTosaRequirementsForBinaryOps(
    ConversionPatternRewriter &rewriter, Operation *op, OpAdaptorT adaptor,
    Type resultType) {
  Value lhs = adaptor.A();
  auto lhsType = lhs.getType().dyn_cast<TensorType>();

  Value rhs = adaptor.B();
  auto rhsType = rhs.getType().dyn_cast<TensorType>();

  auto resultTensorType = resultType.dyn_cast<TensorType>();
  if (!lhsType || !rhsType || !resultTensorType) {
    return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
  }

  Type resultElementType = resultTensorType.getElementType();

  if (!resultElementType.isIntOrFloat()) {
    return rewriter.notifyMatchFailure(op, "only int and float are supported");
  }

  return success();
}

// Element-wise unary ops lowering to TOSA dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOpONNX, typename ElementwiseUnaryOpTOSA,
    typename InputType, typename OutputType>
class ONNXElementwiseUnaryOpLoweringToTOSA
    : public OpConversionPattern<ElementwiseUnaryOpONNX> {
public:
  using OpConversionPattern<ElementwiseUnaryOpONNX>::OpConversionPattern;
  using OpAdaptor = typename ElementwiseUnaryOpONNX::Adaptor;
  LogicalResult matchAndRewrite(ElementwiseUnaryOpONNX op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = *adaptor.getODSOperands(0).begin();
    auto inputType = input.getType().dyn_cast<TensorType>();
    Value output = op.getResult();
    auto outputType = output.getType().dyn_cast<TensorType>();

    if (!inputType || !outputType) {
      return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
    }

    Type inputElementType = inputType.getElementType();
    Type outputElementType = outputType.getElementType();

    if (failed(InputType::checkType(rewriter, inputElementType, op)))
      return rewriter.notifyMatchFailure(op, "input type not supported");

    if (failed(InputType::checkType(rewriter, outputElementType, op)))
      return rewriter.notifyMatchFailure(op, "output type not supported");

    rewriter.replaceOpWithNewOp<ElementwiseUnaryOpTOSA>(
        op, op.getType(), *adaptor.getODSOperands(0).begin());
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

    if (failed(checkBasicTosaRequirementsForBinaryOps<OpAdaptor>(
            rewriter, op, adaptor, op.getResult().getType())))
      return failure();

    Value lhs = adaptor.A();
    Value rhs = adaptor.B();

    rewriter.replaceOpWithNewOp<TosaOpT>(op, op.getType(), lhs, rhs);

    return success();
  }
};

class ONNXMulOpLoweringToTosa : public OpConversionPattern<ONNXMulOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(checkBasicTosaRequirementsForBinaryOps<OpAdaptor>(
            rewriter, op, adaptor, op.getResult().getType())))
      return failure();

    Value lhs = adaptor.A();
    Value rhs = adaptor.B();

    rewriter.replaceOpWithNewOp<mlir::tosa::MulOp>(
        op, op.getType(), lhs, rhs, /*shift =*/0);

    return success();
  }
};

class ONNXReluOpLoweringToTOSA : public OpConversionPattern<ONNXReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.X();

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

static void populateLoweringONNXElementwiseBinaryTemplateOpToTOSAPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXAddOp, mlir::tosa::AddOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXSubOp, mlir::tosa::SubOp>>(
      typeConverter, ctx);
}

static void populateLoweringONNXElementwiseUnaryTemplateOpToTOSAPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseUnaryOpLoweringToTOSA<ONNXNegOp,
                      mlir::tosa::NegateOp, IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXCeilOp, mlir::tosa::CeilOp,
          IsFloat, IsFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXFloorOp, mlir::tosa::FloorOp,
          IsFloat, IsFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXExpOp, mlir::tosa::ExpOp,
          IsFloat, IsFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXLogOp, mlir::tosa::LogOp,
          IsFloat, IsFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXReciprocalOp,
          mlir::tosa::ReciprocalOp, IsFloat, IsFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXTanhOp, mlir::tosa::TanhOp,
          IsFloat, IsFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXSigmoidOp, mlir::tosa::SigmoidOp,
          IsFloat, IsFloat>>(typeConverter, ctx);
}

void populateLoweringONNXElementwiseOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXReluOpLoweringToTOSA, ONNXLeakyReluOpLoweringToTOSA,
      ONNXMulOpLoweringToTosa>(typeConverter, ctx);

  populateLoweringONNXElementwiseBinaryTemplateOpToTOSAPattern(
      patterns, typeConverter, ctx);
  populateLoweringONNXElementwiseUnaryTemplateOpToTOSAPattern(
      patterns, typeConverter, ctx);
}

} // namespace onnx_mlir
