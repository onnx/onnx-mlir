/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Conv2D.cpp - Conv2D Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX conv operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXConvOpLoweringToTOSA : public OpConversionPattern<ONNXConvOp> {
public:
  using OpConversionPattern<ONNXConvOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXConvOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.X();
    auto weights = adaptor.W();
    auto bias = adaptor.B();

    auto inputType = input.getType().cast<ShapedType>();
    auto weightType = weights.getType().cast<ShapedType>();

    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();

    bool biasIsNone = bias.getType().isa<mlir::NoneType>();

    StringAttr autopad = adaptor.auto_padAttr();
    ArrayAttr dilations = adaptor.dilationsAttr();
    IntegerAttr group = adaptor.groupAttr();
    ArrayAttr kernel_shape = adaptor.kernel_shapeAttr();
    ArrayAttr pads = adaptor.padsAttr();
    ArrayAttr strides = adaptor.stridesAttr();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    if (group.getSInt() != 1) {
      return rewriter.notifyMatchFailure(
          op, "ONNX Conv grouping not supported");
    }

    // NOTE: we would like if inferShapes() had filled in explicit padding
    // but currently inferShapes() does not do this for ConvOp (it does for
    // ConvTransposeOp). We have not implemented code for autopad so fail.
    if (autopad && autopad != "NOTSET")
      return op.emitError("padding must be explicit");

    // Convert input [N,C,H,W] -> [N,H,W,C]
    // Create permutation const for input
    SmallVector<int64_t> permVector{0, 2, 3, 1};
    DenseElementsAttr permAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4}, rewriter.getI64Type()), permVector);
    Value permList = tosa::CreateOpAndInfer<tosa::ConstOp>(
        rewriter, op->getLoc(), permAttr.getType(), permAttr);

    // calculate new shape
    SmallVector<int64_t> newInputShape{
        inputShape[0], inputShape[2], inputShape[3], inputShape[1]};

    // get new input type
    Type newInputTy =
        RankedTensorType::get({-1, -1, -1, -1}, inputType.getElementType());

    // create transpose for input
    Value newInput = tosa::CreateOpAndInfer<tosa::TransposeOp>(
        rewriter, op->getLoc(), newInputTy, input, permList);

    // Convert weights [M,C,H,W] -> [M,H,W,C]
    // Create permutation const for input
    SmallVector<int64_t> permWeightVector{0, 2, 3, 1};
    DenseElementsAttr permWeightAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4}, rewriter.getI64Type()), permWeightVector);
    Value permWeightList = tosa::CreateOpAndInfer<tosa::ConstOp>(
        rewriter, op->getLoc(), permWeightAttr.getType(), permWeightAttr);

    // calculate new shape
    SmallVector<int64_t, 4> newWeightShape{
        weightShape[0], weightShape[2], weightShape[3], weightShape[1]};

    // get new weight type
    Type newWeightTy =
        RankedTensorType::get({-1, -1, -1, -1}, weightType.getElementType());

    // create transpose for weight
    Value newWeight = tosa::CreateOpAndInfer<tosa::TransposeOp>(
        rewriter, op->getLoc(), newWeightTy, weights, permWeightList);

    Value newBias = NULL;
    if (bias.getType().isa<NoneType>()) {
      DenseElementsAttr newBiasAttr = DenseElementsAttr::get(
          RankedTensorType::get({weightShape[0]}, rewriter.getF32Type()),
          {0.0F});
      newBias = rewriter.create<tosa::ConstOp>(
          op->getLoc(), newBiasAttr.getType(), newBiasAttr);
    } else {
      newBias = bias;
    }

    ArrayAttr newDilations = NULL;
    if (!dilations) {
      newDilations = rewriter.getI64ArrayAttr({1, 1});
    } else {
      newDilations = dilations;
    }

    ArrayAttr newStrides = NULL;
    if (!strides) {
      newStrides = rewriter.getI64ArrayAttr({1, 1});
    } else {
      newStrides = strides;
    }
    ArrayAttr newPads = NULL;
    if (!pads) {
      newPads = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    } else {
      llvm::SmallVector<int64_t, 4> newPadVec = extractFromI64ArrayAttr(pads);
      newPads = rewriter.getI64ArrayAttr(
          {newPadVec[0], newPadVec[2], newPadVec[3], newPadVec[1]});
    }

    auto oldOutputShape = resultType.cast<ShapedType>().getShape();
    SmallVector<int64_t, 4> newOutputShape{oldOutputShape[0], oldOutputShape[2],
        oldOutputShape[3], oldOutputShape[1]};
    Type newOutputType = RankedTensorType::get(
        {-1, -1, -1, -1}, resultType.cast<ShapedType>().getElementType());

    Value conv2D = tosa::CreateOpAndInfer<tosa::Conv2DOp>(rewriter,
        op->getLoc(), newOutputType, newInput, newWeight, newBias, newPads,
        newStrides, newDilations);

    // Convert weights [M,C,H,W] -> [M,H,W,C]
    // Create permutation const for input
    SmallVector<int64_t> permOutputVector{0, 3, 1, 2};
    DenseElementsAttr permOutputAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4}, rewriter.getI64Type()), permOutputVector);
    Value permOutputList = rewriter.create<tosa::ConstOp>(
        op->getLoc(), permOutputAttr.getType(), permOutputAttr);

    auto outputShape = conv2D.getType().cast<ShapedType>().getShape();
    // calculate new shape
    SmallVector<int64_t, 4> newOutputShapeReturn{
        outputShape[0], outputShape[3], outputShape[1], outputShape[2]};

    // get new weight type
    Type newOutputTy = RankedTensorType::get(
        {-1, -1, -1, -1}, resultType.cast<ShapedType>().getElementType());

    // create transpose for weight
    tosa::CreateReplaceOpAndInfer<tosa::TransposeOp>(
        rewriter, op, newOutputTy, conv2D, permOutputList);

    return success();
  }
};
} // namespace

void populateLoweringONNXConvOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir