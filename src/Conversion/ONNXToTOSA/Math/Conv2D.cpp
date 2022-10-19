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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
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

    if (group.getSInt() != 1) {
      rewriter.notifyMatchFailure(op, "ONNX Conv grouping not supported");
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
    Value permList = rewriter.create<tosa::ConstOp>(
        op->getLoc(), permAttr.getType(), permAttr);

    // calculate new shape
    SmallVector<int64_t> newInputShape{
        inputShape[0], inputShape[2], inputShape[3], inputShape[1]};

    // get new input type
    Type newInputTy =
        RankedTensorType::get(newInputShape, inputType.getElementType());

    // create transpose for input
    rewriter.create<tosa::TransposeOp>(
        op->getLoc(), newInputTy, input, permList);

    // Convert weights [M,C,H,W] -> [M,H,W,C]
    // Create permutation const for input
    SmallVector<int64_t> permWeightVector{0, 2, 3, 1};
    DenseElementsAttr permWeightAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4}, rewriter.getI64Type()), permWeightVector);
    Value permWeightList = rewriter.create<tosa::ConstOp>(
        op->getLoc(), permWeightAttr.getType(), permWeightAttr);

    // calculate new shape
    SmallVector<int64_t> newWeightShape{
        weightShape[0], weightShape[2], weightShape[3], weightShape[1]};

    // get new weight type
    Type newWeightTy =
        RankedTensorType::get(newWeightShape, weightType.getElementType());

    // create transpose for weight
    rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
        op, newWeightTy, weights, permWeightList);

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