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
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

/// When setting the groups parameters, we have to create multiple conv2d ops
/// where the input, kernel and bias is a slice of the original inputs.
/// Afterwards we have to concat the results into a single tensor
Value createConvInGroups(PatternRewriter &rewriter, Operation *op,
    Type &resultType, const llvm::ArrayRef<int64_t> weightShape,
    Value &newInput, Value &newWeight, Value &bias, const int64_t groups,
    ArrayAttr &pads, ArrayAttr &strides, ArrayAttr &dilations) {
  // Set up constants outside of loop
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

class ONNXConvOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXConvOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXConvOp::getOperationName(), 1, ctx) {}

  using OpAdaptor = typename ONNXConvOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    OpAdaptor adaptor(operands, op->getAttrDictionary());
    auto convOp = llvm::cast<ONNXConvOp>(op);

    auto input = adaptor.X();
    auto weights = adaptor.W();
    auto bias = adaptor.B();

    auto inputType = input.getType().cast<TensorType>();
    auto weightType = weights.getType().cast<ShapedType>();

    // Get shapehelper for autopad attributes
    IndexExprBuilderForTosa createTosaIE(rewriter, convOp->getLoc());
    NewONNXConvOpShapeHelper shapeHelper(op, operands, &createTosaIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    auto weightShape = weightType.getShape();

    Type resultType = convOp.getResult().getType();

    if (inputType.getShape().size() != 4) {
      return rewriter.notifyMatchFailure(convOp, "TOSA only supports conv 2d");
    }

    // Convert input [N,IC,IH,IW] -> [N,IH,IW,IC]
    Value newInput =
        tosa::createTosaTransposedTensor(rewriter, convOp, input, {0, 2, 3, 1});

    // Convert weights [OC,IC,KH,KW] -> [OC,KH,KW,IC]
    Value newWeight = tosa::createTosaTransposedTensor(
        rewriter, convOp, weights, {0, 2, 3, 1});

    if (bias.getType().isa<NoneType>()) {
      DenseElementsAttr newBiasAttr = DenseElementsAttr::get(
          RankedTensorType::get({weightShape[0]}, rewriter.getF32Type()),
          {0.0F});
      bias = rewriter.create<mlir::tosa::ConstOp>(
          convOp->getLoc(), newBiasAttr.getType(), newBiasAttr);
    }

    ArrayAttr dilations = rewriter.getI64ArrayAttr(shapeHelper.dilations);
    ArrayAttr strides = rewriter.getI64ArrayAttr(shapeHelper.strides);
    llvm::SmallVector<int64_t, 4> transposedPads =
        tosa::createInt64VectorFromIndexExpr(shapeHelper.pads);
    // reorder padding values
    ArrayAttr pads = rewriter.getI64ArrayAttr({transposedPads[0],
        transposedPads[2], transposedPads[1], transposedPads[3]});

    // Handle group parameter by creating multiple convs
    const int64_t group = adaptor.group();
    Value conv2D = NULL;
    if (group == 1) {
      Type newConvOutputType = RankedTensorType::get(
          {-1, -1, -1, -1}, resultType.cast<ShapedType>().getElementType());

      conv2D = tosa::CreateOpAndInfer<mlir::tosa::Conv2DOp>(rewriter,
          convOp->getLoc(), newConvOutputType, newInput, newWeight, bias, pads,
          strides, dilations);
    } else {
      conv2D = createConvInGroups(rewriter, convOp, resultType, weightShape,
          newInput, newWeight, bias, group, pads, strides, dilations);
    }

    // Convert output [N,OH,OW,OC] -> [N,OC,OH,OW]
    Value newOutput = tosa::createTosaTransposedTensor(
        rewriter, convOp, conv2D, {0, 3, 1, 2});

    rewriter.replaceOp(convOp, {newOutput});
    return success();
  }
};
} // namespace

void populateLoweringONNXConvOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir