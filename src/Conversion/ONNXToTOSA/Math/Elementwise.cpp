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
    rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(op, op.getType(), input,
        rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max()),
        rewriter.getF32FloatAttr(0.0f),
        rewriter.getF32FloatAttr(std::numeric_limits<float>::max()));
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
      ONNXFloorOpLoweringToTOSA, ONNXReluOpLoweringToTOSA,
      ONNXDivOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
