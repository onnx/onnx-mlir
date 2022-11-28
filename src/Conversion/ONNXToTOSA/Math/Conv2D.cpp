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
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/PatternMatch.h>

using namespace mlir;

namespace onnx_mlir {

namespace {

/// When setting the groups parameters, we have to create multiple conv2d ops
/// where the input, kernel and bias is a slice of the original inputs.
/// Afterwards we have to concat the results into a single tensor
Value createConvInGroups(PatternRewriter &rewriter, Operation *op,
    Type &resultType, const llvm::ArrayRef<int64_t> weightShape,
    Value &newInput, Value &newWeight, Value &bias, IntegerAttr &group,
    ArrayAttr &pads, ArrayAttr &strides, ArrayAttr &dilations) {
  // Set up constants outside of loop
  const int64_t groups = group.getSInt();
  const int64_t sizeOfSliceInput = weightShape[1];
  const int64_t sizeOfSliceKernel = weightShape[0] / groups;
  auto newInputShape = newInput.getType().cast<ShapedType>().getShape();

  llvm::SmallVector<int64_t, 4> inputSize = {
      newInputShape[0], newInputShape[1], newInputShape[2], sizeOfSliceInput};
  llvm::SmallVector<int64_t, 4> kernelSize = {
      sizeOfSliceKernel, weightShape[2], weightShape[3], weightShape[1]};
  llvm::SmallVector<Value> sliceValues;

  for (int64_t i = 0; i < groups; i++) {
    // Slice input
    Value newSliceInput = tosa::sliceTensor(
        rewriter, op, newInput, inputSize, {0, 0, 0, i * sizeOfSliceInput});

    // Slice kernel
    Value newSliceWeight = tosa::sliceTensor(
        rewriter, op, newWeight, kernelSize, {i * sizeOfSliceKernel, 0, 0, 0});

    // Slice bias
    Value newSliceBias = tosa::sliceTensor(
        rewriter, op, bias, {sizeOfSliceKernel}, {i * sizeOfSliceKernel});

    // Create conv
    Type newConvOutputType = RankedTensorType::get(
        {-1, -1, -1, -1}, resultType.cast<ShapedType>().getElementType());
    Value tempConv2D = tosa::CreateOpAndInfer<mlir::tosa::Conv2DOp>(rewriter,
        op->getLoc(), newConvOutputType, newSliceInput, newSliceWeight,
        newSliceBias, pads, strides, dilations);
    // Add value to vector
    sliceValues.push_back(tempConv2D);
  }
  // Create concat op
  Type newConcatOutputType = RankedTensorType::get(
      {-1, -1, -1, -1}, resultType.cast<ShapedType>().getElementType());
  Value conv2D = tosa::CreateOpAndInfer<mlir::tosa::ConcatOp>(
      rewriter, op->getLoc(), newConcatOutputType, sliceValues, 3);
  return conv2D;
}

class ONNXConvOpLoweringToTOSA : public OpConversionPattern<ONNXConvOp> {
public:
  using OpConversionPattern<ONNXConvOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXConvOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.X();
    auto weights = adaptor.W();
    auto bias = adaptor.B();

    auto weightType = weights.getType().cast<ShapedType>();

    auto weightShape = weightType.getShape();

    StringAttr autopad = adaptor.auto_padAttr();
    ArrayAttr dilations = adaptor.dilationsAttr();
    IntegerAttr group = adaptor.groupAttr();
    ArrayAttr pads = adaptor.padsAttr();
    ArrayAttr strides = adaptor.stridesAttr();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    // NOTE: we would like if inferShapes() had filled in explicit padding
    // but currently inferShapes() does not do this for ConvOp (it does for
    // ConvTransposeOp). We have not implemented code for autopad so fail.
    if (autopad && autopad != "NOTSET")
      return rewriter.notifyMatchFailure(op, "padding must be explicit");

    // Convert input [N,IC,IH,IW] -> [N,IH,IW,IC]
    Value newInput =
        tosa::createTosaTransposedTensor(rewriter, op, input, {0, 2, 3, 1});

    // Convert weights [OC,IC,KH,KW] -> [OC,KH,KW,IC]
    Value newWeight =
        tosa::createTosaTransposedTensor(rewriter, op, weights, {0, 2, 3, 1});

    if (bias.getType().isa<NoneType>()) {
      DenseElementsAttr newBiasAttr = DenseElementsAttr::get(
          RankedTensorType::get({weightShape[0]}, rewriter.getF32Type()),
          {0.0F});
      bias = rewriter.create<mlir::tosa::ConstOp>(
          op->getLoc(), newBiasAttr.getType(), newBiasAttr);
    }
    if (!dilations) {
      dilations = rewriter.getI64ArrayAttr({1, 1});
    }
    if (!strides) {
      strides = rewriter.getI64ArrayAttr({1, 1});
    }
    if (!pads) {
      pads = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    } else {
      llvm::SmallVector<int64_t, 4> newPadVec = extractFromI64ArrayAttr(pads);
      pads = rewriter.getI64ArrayAttr(
          {newPadVec[0], newPadVec[2], newPadVec[1], newPadVec[3]});
    }

    Value conv2D = NULL;
    if (group.getSInt() == 1) {
      Type newConvOutputType = RankedTensorType::get(
          {-1, -1, -1, -1}, resultType.cast<ShapedType>().getElementType());

      conv2D = tosa::CreateOpAndInfer<mlir::tosa::Conv2DOp>(rewriter,
          op->getLoc(), newConvOutputType, newInput, newWeight, bias, pads,
          strides, dilations);
    } else {
      conv2D = createConvInGroups(rewriter, op, resultType, weightShape,
          newInput, newWeight, bias, group, pads, strides, dilations);
    }

    // Convert output [N,OH,OW,OC] -> [N,OC,OH,OW]
    Value newOutput =
        tosa::createTosaTransposedTensor(rewriter, op, conv2D, {0, 3, 1, 2});

    rewriter.replaceOp(op, {newOutput});
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