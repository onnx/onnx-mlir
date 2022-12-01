/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- MaxPoolSingleOut.cpp - ONNX Op Transform --------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===-----------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;

// ONNX MaxPool operation
//
// ONNX MaxPool operation with a single output.
// See ONNXMaxPoolOp for a full description of the MaxPool semantics.
//
// Attributes:
//  auto_pad	::mlir::StringAttr	string attribute
//  ceil_mode	::mlir::IntegerAttr	64-bit signed integer attribute
//  dilations	::mlir::ArrayAttr	64-bit integer array attribute
//  kernel_shape  ::mlir::ArrayAttr	64-bit integer array attribute
//  pads	  ::mlir::ArrayAttr	64-bit integer array attribute
//  storage_order ::mlir::IntegerAttr	64-bit signed integer attribute
// strides	  ::mlir::ArrayAttr	64-bit integer array attribute
//
// Operands:
// X	memref of any type values or tensor of any type values
//
// Results:
// o_Y	memref of any type values or tensor of any type values
//
class ONNXMaxPoolSingleOutOpToTorchLowering
    : public OpConversionPattern<ONNXMaxPoolSingleOutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    Value x = adaptor.X();
    ArrayAttr kernelShape = adaptor.kernel_shapeAttr();
    ArrayAttr dilations = adaptor.dilationsAttr();
    ArrayAttr pads = adaptor.padsAttr();
    ArrayAttr strides = adaptor.stridesAttr();
    int64_t ceilingMode = adaptor.ceil_mode();
    IntegerAttr ceilingModeAttr = adaptor.ceil_modeAttr();
    IntegerType intType = IntegerType::get(context, 64);
    mlir::FloatType floatType = mlir::FloatType::getF64(context);

    // Get mlir attributes as vectors
    dim_pads padsOnnxList =
        createPadsArrayAttribute(pads, intType, loc, rewriter);
    // Dilation has a default value of 1
    std::vector<Value> dilationOnnxList =
        createArrayAttribute(dilations, intType, loc, rewriter, 1);
    std::vector<Value> kernelShapeOnnxList;
    std::vector<Value> stridesOnnxList;

    if (kernelShape) {
      for (unsigned i = 0; i < kernelShape.size(); i++) {
        auto kernelShapeElement = IntegerAttr::get(
            intType, kernelShape[i].cast<IntegerAttr>().getValue());
        Value kernelShapeConstInt =
            rewriter.create<ConstantIntOp>(loc, kernelShapeElement);
        kernelShapeOnnxList.push_back(kernelShapeConstInt);
      }
    }

    if (strides) {
      for (unsigned i = 0; i < strides.size(); i++) {
        auto strideElement = IntegerAttr::get(
            intType, strides[i].cast<IntegerAttr>().getValue());
        Value strideElementConstInt =
            rewriter.create<ConstantIntOp>(loc, strideElement);
        stridesOnnxList.push_back(strideElementConstInt);
      }
    }

    // If ceilingMode is 0 (default) use floor rounding when computing the
    // output shape, else use ceil.
    Value constBoolOpValue = rewriter.create<ConstantBoolOp>(loc, false);
    Value ceilingModeVal;
    if (ceilingModeAttr) {
      if (ceilingMode == 0)
        ceilingModeVal = rewriter.create<ConstantBoolOp>(loc, false);
      else
        ceilingModeVal = rewriter.create<ConstantBoolOp>(loc, true);
    } else
      ceilingModeVal = constBoolOpValue;

    // Create maxpool mlir values
    Value stridesList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{stridesOnnxList});
    Value padsList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{padsOnnxList.padding});
    Value dilationList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{dilationOnnxList});
    Value kernelShapeList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{kernelShapeOnnxList});

    // Determine input and result type
    TensorType opTensorType = op.getResult().getType().cast<TensorType>();
    auto resultType = Torch::ValueTensorType::get(
        context, opTensorType.getShape(), opTensorType.getElementType());

    // Allow symmetric padding and create additional padding op to support
    // asymmetric padding in `torch-mlir`
    Value result;
    if (!padsOnnxList.isSymmetric) {
      std::vector<int64_t> padShape =
          x.getType().cast<Torch::ValueTensorType>().getSizes();
      for (unsigned i = 0; i < pads.size() / 2; i++) {
        llvm::APInt startDim = pads[i].cast<IntegerAttr>().getValue();
        llvm::APInt endDim =
            pads[i + (pads.size() / 2)].cast<IntegerAttr>().getValue();
        padShape[i + 2] += (startDim + endDim).getZExtValue();
      }
      auto padType =
          Torch::ValueTensorType::get(context, llvm::makeArrayRef(padShape),
              x.getType().cast<Torch::ValueTensorType>().getDtype());

      // Construct zero padding op since `torch` does not support asymmetric
      // padding for maxpool2d
      float negInf = -std::numeric_limits<float>::max();
      FloatAttr negInfAttr = FloatAttr::get(floatType, negInf);
      Value zeroPad = rewriter.create<ConstantFloatOp>(loc, negInfAttr);
      Value padTensor = rewriter.create<AtenConstantPadNdOp>(
          loc, padType, x, padsList, zeroPad);
      setLayerNameAttr(op, padTensor.getDefiningOp());

      IntegerAttr zeroIntAttr = IntegerAttr::get(intType, 0);
      Value padValue = rewriter.create<ConstantIntOp>(loc, zeroIntAttr);
      Value zeroPadsList = rewriter.create<PrimListConstructOp>(loc,
          Torch::ListType::get(rewriter.getType<Torch::IntType>()),
          ValueRange{padValue, padValue});

      result = rewriter.create<AtenMaxPool2dOp>(loc, resultType, padTensor,
          kernelShapeList, stridesList, zeroPadsList, dilationList,
          ceilingModeVal);
    } else {
      result = rewriter.create<AtenMaxPool2dOp>(loc, resultType, x,
          kernelShapeList, stridesList, padsList, dilationList, ceilingModeVal);
    }
    setLayerNameAttr(op, result.getDefiningOp());

    rewriter.replaceOpWithNewOp<TorchConversion::ToBuiltinTensorOp>(
        op, op.getResult().getType(), result);
    return success();
  }
};

void populateLoweringONNXToTorchMaxPoolSingleOutOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMaxPoolSingleOutOpToTorchLowering>(typeConverter, ctx);
}
