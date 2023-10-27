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
#include "mlir/IR/Matchers.h"
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
  Value lhs = adaptor.getA();
  auto lhsType = lhs.getType().dyn_cast<TensorType>();

  Value rhs = adaptor.getB();
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
      return failure();

    if (failed(InputType::checkType(rewriter, outputElementType, op)))
      return failure();

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

    auto loc = op.getLoc();
    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();

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

class ONNXMulOpLoweringToTosa : public OpConversionPattern<ONNXMulOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(checkBasicTosaRequirementsForBinaryOps<OpAdaptor>(
            rewriter, op, adaptor, op.getResult().getType())))
      return failure();

    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();

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
  Value constZero = tosaBuilder.getSplattedConst(0.0, outputType.getShape());

  auto mul = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(rewriter, op->getLoc(),
      outputType, input,
      tosaBuilder.getSplattedConst(alpha, outputType.getShape()),
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
    FloatAttr alphaAttr = adaptor.getAlphaAttr();
    if (alphaAttr) {
      // No easy interface in MLIR to get value as float
      alpha = alphaAttr.getValueAsDouble();
    }
    return LegalizeFloatingPointPrelu(
        op, rewriter, adaptor.getX(), alpha, outputType);
  }
};

class ONNXClipOpLoweringToTOSA : public OpConversionPattern<ONNXClipOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = ONNXClipOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXClipOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto res = adaptor.getInput();
    auto min = adaptor.getMin();
    auto max = adaptor.getMax();

    auto matchIntOrFloat = [&](Value val) -> std::tuple<bool, int64_t, float> {
      APInt valueInt(64, 0);
      APFloat valueFloat(0.0f);
      if (matchPattern(val, m_ConstantInt(&valueInt))) {
        auto intVal = valueInt.getSExtValue();
        return {true, intVal, static_cast<float>(intVal)};
      }
      if (matchPattern(val, m_ConstantFloat(&valueFloat))) {
        float floatVal = valueFloat.convertToFloat();
        return {true, static_cast<int64_t>(floatVal), floatVal};
      }
      return {false, 0, 0.0};
    };

    // Use ClampOp if min and max are splat constants.
    // Otherwise, MaximumOp and MinimumOp to clamp min and max, respectively.
    auto [isSplatConstMin, minInt, minFloat] = matchIntOrFloat(min);
    auto [isSplatConstMax, maxInt, maxFloat] = matchIntOrFloat(max);
    if (isSplatConstMin && isSplatConstMax) {
      rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(op, op.getType(), res,
          rewriter.getI64IntegerAttr(minInt),
          rewriter.getI64IntegerAttr(maxInt),
          rewriter.getF32FloatAttr(minFloat),
          rewriter.getF32FloatAttr(maxFloat));
    } else {
      if (!isNoneValue(min)) {
        res = tosa::CreateOpAndInfer<mlir::tosa::MaximumOp>(
            rewriter, op->getLoc(), op.getType(), res, min);
      }
      if (!isNoneValue(max)) {
        res = tosa::CreateOpAndInfer<mlir::tosa::MinimumOp>(
            rewriter, op->getLoc(), op.getType(), res, max);
      }
      rewriter.replaceOp(op, res);
    }
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
    auto resultType = op.getResult().getType().template cast<TensorType>();
    Type resultElementType = resultType.getElementType();

    TosaBuilder tosaBuilder(rewriter, op->getLoc());

    if (resultElementType.isSignlessInteger(32)) {
      // tosa::DivOp takes 32-but signless integers as inputs
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
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXFloorOp, mlir::tosa::FloorOp,
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXExpOp, mlir::tosa::ExpOp,
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXLogOp, mlir::tosa::LogOp,
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXReciprocalOp,
          mlir::tosa::ReciprocalOp, IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXTanhOp, mlir::tosa::TanhOp,
          IsIntOrFloat, IsIntOrFloat>,
      ONNXElementwiseUnaryOpLoweringToTOSA<ONNXSigmoidOp, mlir::tosa::SigmoidOp,
          IsIntOrFloat, IsIntOrFloat>>(typeConverter, ctx);
}

void populateLoweringONNXElementwiseOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXReluOpLoweringToTOSA, ONNXLeakyReluOpLoweringToTOSA,
      ONNXMulOpLoweringToTosa, ONNXClipOpLoweringToTOSA,
      ONNXDivOpLoweringToTOSA>(typeConverter, ctx);

  populateLoweringONNXElementwiseBinaryTemplateOpToTOSAPattern(
      patterns, typeConverter, ctx);
  populateLoweringONNXElementwiseUnaryTemplateOpToTOSAPattern(
      patterns, typeConverter, ctx);
}

} // namespace onnx_mlir
