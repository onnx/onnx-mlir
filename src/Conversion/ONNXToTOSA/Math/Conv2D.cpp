/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Conv2D.cpp - Conv2D Op ------------------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX conv operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

/// tosa.conv2d does not support dividing the channels into groups.
/// When the onnx operator requires this, we have to create multiple ops
/// where the input, kernel and bias is a slice of the original inputs.
/// Afterwards we have to concat the results of these into a single tensor.
Value createConvInGroups(PatternRewriter &rewriter, Operation *op,
    TosaBuilder &tosaBuilder, Type &resultType,
    const llvm::ArrayRef<int64_t> weightShape, Value &newInput,
    Value &newWeight, Value &bias, const int64_t groups,
    DenseI64ArrayAttr &pads, DenseI64ArrayAttr &strides,
    DenseI64ArrayAttr &dilations) {
  // Set up constants outside of loop
  const int64_t sizeOfSliceInput = weightShape[1];
  const int64_t sizeOfSliceKernel = weightShape[0] / groups;
  auto newInputShape = mlir::cast<ShapedType>(newInput.getType()).getShape();

  llvm::SmallVector<int64_t, 4> inputSize = {
      newInputShape[0], newInputShape[1], newInputShape[2], sizeOfSliceInput};
  llvm::SmallVector<int64_t, 4> kernelSize = {
      sizeOfSliceKernel, weightShape[2], weightShape[3], weightShape[1]};
  llvm::SmallVector<Value> sliceValues;

  for (int64_t i = 0; i < groups; i++) {
    // Slice input
    Value newSliceInput =
        tosaBuilder.slice(newInput, inputSize, {0, 0, 0, i * sizeOfSliceInput});

    // Slice kernel
    Type newSliceWeightType = RankedTensorType::get(kernelSize,
        mlir::cast<ShapedType>(newWeight.getType()).getElementType());
    Value newSliceWeight = rewriter.create<tensor::ExtractSliceOp>(
        newWeight.getLoc(), newSliceWeightType, newWeight, ValueRange({}),
        SmallVector<Value>{}, ValueRange({}),
        rewriter.getDenseI64ArrayAttr({i * sizeOfSliceKernel, 0, 0, 0}),
        rewriter.getDenseI64ArrayAttr(kernelSize),
        rewriter.getDenseI64ArrayAttr({1, 1, 1, 1}));

    // Slice bias
    Type newSliceBiasType = RankedTensorType::get({sizeOfSliceKernel},
        mlir::cast<ShapedType>(bias.getType()).getElementType());
    Value newSliceBias = rewriter.create<tensor::ExtractSliceOp>(bias.getLoc(),
        newSliceBiasType, bias, ValueRange({}), SmallVector<Value>{},
        ValueRange({}), rewriter.getDenseI64ArrayAttr({i * sizeOfSliceKernel}),
        rewriter.getDenseI64ArrayAttr({sizeOfSliceKernel}),
        rewriter.getDenseI64ArrayAttr({1}));

    // Create conv
    Type newConvOutputType = RankedTensorType::get(
        llvm::SmallVector<int64_t, 4>(4, ShapedType::kDynamic),
        mlir::cast<ShapedType>(resultType).getElementType());
    Value tempConv2D = tosa::CreateOpAndInfer<mlir::tosa::Conv2DOp>(rewriter,
        op->getLoc(), newConvOutputType, newSliceInput, newSliceWeight,
        newSliceBias, pads, strides, dilations);
    // Add value to vector
    sliceValues.push_back(tempConv2D);
  }
  // Create concat op
  Type newConcatOutputType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(4, ShapedType::kDynamic),
      mlir::cast<ShapedType>(resultType).getElementType());
  Value conv2D = tosa::CreateOpAndInfer<mlir::tosa::ConcatOp>(
      rewriter, op->getLoc(), newConcatOutputType, sliceValues, 3);
  return conv2D;
}

class ONNXConvOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXConvOpLoweringToTOSA(MLIRContext *ctx, int64_t groupedConvThreshold)
      : ConversionPattern(ONNXConvOp::getOperationName(), 1, ctx),
        groupedConvThreshold(groupedConvThreshold) {}

  using OpAdaptor = typename ONNXConvOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    OpAdaptor adaptor(operands, op->getAttrDictionary());
    auto loc = op->getLoc();
    auto convOp = llvm::cast<ONNXConvOp>(op);

    TosaBuilder tosaBuilder(rewriter, loc);

    auto input = adaptor.getX();
    auto weights = adaptor.getW();
    auto bias = adaptor.getB();

    auto inputType = mlir::cast<TensorType>(input.getType());
    auto weightType = mlir::cast<ShapedType>(weights.getType());

    if (!inputType || !weightType || !inputType.hasStaticShape() ||
        !weightType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "only ranked tensor types are supported");
    }

    // Get shapehelper for autopad attributes
    IndexExprBuilderForTosa createTosaIE(rewriter, convOp->getLoc());
    ONNXConvOpShapeHelper shapeHelper(op, operands, &createTosaIE);
    if (shapeHelper.computeShape().failed()) {
      return rewriter.notifyMatchFailure(convOp, "Could not infer shapes");
    }

    auto weightShape = weightType.getShape();

    Type resultType = convOp.getResult().getType();

    if (inputType.getShape().size() != 4) {
      return rewriter.notifyMatchFailure(
          convOp, "only 2d tensor support is implemented");
    }

    // Convert input [N,IC,IH,IW] -> [N,IH,IW,IC]
    Value newInput = tosaBuilder.transpose(input, {0, 2, 3, 1});

    // Convert weights [OC,IC,KH,KW] -> [OC,KH,KW,IC]
    Value newWeight = tosaBuilder.transpose(weights, {0, 2, 3, 1});

    if (mlir::isa<NoneType>(bias.getType())) {
      DenseElementsAttr newBiasAttr = DenseElementsAttr::get(
          RankedTensorType::get({weightShape[0]}, rewriter.getF32Type()),
          {0.0F});
      bias = rewriter.create<mlir::tosa::ConstOp>(
          convOp->getLoc(), newBiasAttr.getType(), newBiasAttr);
    }

    DenseI64ArrayAttr dilations =
        rewriter.getDenseI64ArrayAttr(shapeHelper.dilations);
    DenseI64ArrayAttr strides =
        rewriter.getDenseI64ArrayAttr(shapeHelper.strides);

    if (!IndexExpr::isLiteral(shapeHelper.pads))
      return rewriter.notifyMatchFailure(op, "pads is not a literal.");
    llvm::SmallVector<int64_t, 4> pads;
    IndexExpr::getLiteral(shapeHelper.pads, pads);
    // reorder padding values
    llvm::SmallVector<int64_t, 4> reorderedPads = {
        pads[0], pads[2], pads[1], pads[3]};
    FailureOr<Value> resizedInput = tosaBuilder.resizeWindowBasedOps(newInput,
        cast<RankedTensorType>(newInput.getType()).getShape(),
        {weightShape[2], weightShape[3]}, reorderedPads, shapeHelper.strides,
        shapeHelper.dilations);

    if (failed(resizedInput))
      return rewriter.notifyMatchFailure(
          op, "could not resize input to match parameters");

    DenseI64ArrayAttr newPads = rewriter.getDenseI64ArrayAttr(reorderedPads);

    // Handle group parameter by creating multiple convs
    const int64_t group = adaptor.getGroup();
    Value conv2D = NULL;
    if (group == 1) {
      Type newConvOutputType = RankedTensorType::get(
          llvm::SmallVector<int64_t, 4>(4, ShapedType::kDynamic),
          mlir::cast<ShapedType>(resultType).getElementType());

      conv2D = tosa::CreateOpAndInfer<mlir::tosa::Conv2DOp>(rewriter,
          convOp->getLoc(), newConvOutputType, newInput, newWeight, bias,
          newPads, strides, dilations);
    } else {
      auto inputChannels = inputType.getDimSize(1);
      auto outputChannels = cast<ShapedType>(resultType).getDimSize(1);
      if (group == inputChannels && (outputChannels % inputChannels == 0)) {
        // If the group == inputChannels and
        // outputChannels == inputChannels * integerNumber,
        // this grouped convolution is equal to a Depthwise convolution.

        // Convert weights [OC,IC,KH,KW] -> [KH, KW, OC, M(ChannelMultiplier)]
        Value transposedWeight = tosaBuilder.transpose(weights, {2, 3, 0, 1});
        // A reshape op is needed to adhere to the TOSA standard
        // https://www.mlplatform.org/tosa/tosa_spec.html#_depthwise_conv2d
        Value newWeight = tosaBuilder.reshape(
            transposedWeight, {weightShape[2], weightShape[3], inputChannels,
                                  outputChannels / inputChannels});

        Type newConvOutputType = RankedTensorType::get(
            llvm::SmallVector<int64_t, 4>(4, ShapedType::kDynamic),
            cast<ShapedType>(resultType).getElementType());

        conv2D = tosa::CreateOpAndInfer<mlir::tosa::DepthwiseConv2DOp>(rewriter,
            convOp->getLoc(), newConvOutputType, newInput, newWeight, bias,
            newPads, strides, dilations);
      } else if (group <= groupedConvThreshold) {
        // Decompose group convolution into a concatenation of tosa.conv2d ops
        // can be costly, so only allow it when the number of groups is less
        // than configurable threshold.

        conv2D = createConvInGroups(rewriter, convOp, tosaBuilder, resultType,
            weightShape, newInput, newWeight, bias, group, newPads, strides,
            dilations);
      } else {
        return rewriter.notifyMatchFailure(
            op, "this type of grouped Conv is not supported");
      }
    }

    // Convert output [N,OH,OW,OC] -> [N,OC,OH,OW]
    Value newOutput = tosaBuilder.transpose(conv2D, {0, 3, 1, 2});

    rewriter.replaceOp(convOp, {newOutput});
    return success();
  }

private:
  int64_t groupedConvThreshold;
};

} // namespace

void populateLoweringONNXConvOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx,
    int64_t groupedConvThreshold) {
  patterns.insert<ONNXConvOpLoweringToTOSA>(ctx, groupedConvThreshold);
}

} // namespace onnx_mlir