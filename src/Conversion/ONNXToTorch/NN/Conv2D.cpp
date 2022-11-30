/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ Conv2D.cpp - Lowering Convolution Op ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// ========================================================================
//
// This file lowers the ONNX Convolution Operators to Torch dialect.
//
//===-----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

//  ONNX Conv operation
//
//  The convolution operator consumes an input tensor and a filter,
//  and computes the output.
//
//  Attributes:
//    auto_pad	    ::mlir::StringAttr	string attribute
//    dilations	    ::mlir::ArrayAttr	64-bit integer array
//    group		    ::mlir::IntegerAttr	64-bit signed integer
//    kernel_shape	::mlir::ArrayAttr	64-bit integer array
//    pads		    ::mlir::ArrayAttr	64-bit integer array
//    strides		::mlir::ArrayAttr	64-bit integer array
// Operands:
//    X tensor of 16-bit/32-bit/64-bit float values or memref of any type values
//    W tensor of 16-bit/32-bit/64-bit float values or memref of any type values
//    B tensor of 16-bit/32-bit/64-bit float values or memref of any type values
//    or none type
// Results:
//    Y tensor of 16-bit/32-bit/64-bit float values or memref of any type values
//    or none type
//
class ONNXConvOpToTorchLowering : public OpConversionPattern<ONNXConvOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    Value x = adaptor.X();
    Value w = adaptor.W();
    Value b = adaptor.B();
    bool biasIsNone = b.getType().isa<mlir::NoneType>();

    mlir::StringAttr autopad = adaptor.auto_padAttr();
    mlir::ArrayAttr dilations = adaptor.dilationsAttr();
    mlir::IntegerAttr group = adaptor.groupAttr();
    mlir::ArrayAttr kernal_shape = adaptor.kernel_shapeAttr();
    mlir::ArrayAttr pads = adaptor.padsAttr();
    mlir::ArrayAttr strides = adaptor.stridesAttr();

    // NOTE: we would like if inferShapes() had filled in explicit padding
    // but currently inferShapes() does not do this for ConvOp (it does for
    // ConvTransposeOp). We have not implemented code for autopad so fail.
    if (autopad && autopad != "NOTSET")
      return op.emitError("padding must be explicit");

    // Create vector of tensor list iterate through the ArrayAttribute
    // list.
    mlir::IntegerType sintType =
        IntegerType::get(context, 64, IntegerType::SignednessSemantics::Signed);
    dim_pads translatepadsList =
        createPadsArrayAttribute(pads, sintType, loc, rewriter);
    std::vector<Value> dilationonnxList =
        createArrayAttribute(dilations, sintType, loc, rewriter, 1);
    std::vector<Value> kernalshapeonnxList =
        createArrayAttribute(kernal_shape, sintType, loc, rewriter);
    std::vector<Value> stridesonnxList =
        createArrayAttribute(strides, sintType, loc, rewriter);

    // Check that convolution has symmetric padding. Asymmetric padding is
    // currently not supported.
    if (!translatepadsList.isSymmetric)
      return op.emitError(
          "convolutions with asymmetric padding are not supported");

    // If group Value is null, assigning default value.
    Value groupTorchInt;
    if (group) {
      groupTorchInt = rewriter.create<ConstantIntOp>(loc, group);
    } else {
      // NOTE: we would like if inferShapes() had filled in default values
      // so we could assume `group` is always set, but currently inferShapes()
      // does not do this for ConvOp (it does for ConvTransposeOp).
      auto oneAttr = IntegerAttr::get(sintType, 1);
      groupTorchInt = rewriter.create<ConstantIntOp>(loc, oneAttr);
    }

    // Create the Torch List type using above created vectors.
    Value stridesList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{stridesonnxList});
    Value dilationList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{dilationonnxList});
    Value padsList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{translatepadsList.padding});
    Value outputPadsList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{});
    Value transposeVal = rewriter.create<Torch::ConstantBoolOp>(loc, false);

    // Create a tensor types using onnx operands.
    Torch::ValueTensorType xTensorType =
        x.getType().cast<Torch::ValueTensorType>();
    Torch::ValueTensorType wTensorType =
        w.getType().cast<Torch::ValueTensorType>();
    mlir::TensorType opTensorType = op.getResult().getType().cast<TensorType>();

    auto xType = Torch::ValueTensorType::get(
        context, xTensorType.getSizes(), xTensorType.getDtype());
    auto wType = Torch::ValueTensorType::get(
        context, wTensorType.getSizes(), wTensorType.getDtype());
    auto resultType = Torch::ValueTensorType::get(
        context, opTensorType.getShape(), opTensorType.getElementType());

    auto xTorchTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, xType, x);
    auto wTorchTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, wType, w);
    Value bTorchTensor;
    if (biasIsNone) {
      bTorchTensor = rewriter.create<Torch::ConstantNoneOp>(loc);
    } else {
      Torch::ValueTensorType bTensorType =
          b.getType().cast<Torch::ValueTensorType>();
      auto bType = Torch::ValueTensorType::get(
          context, bTensorType.getSizes(), bTensorType.getDtype());
      bTorchTensor =
          rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
              loc, bType, b);
    }

    // Emit the Conv2d operation in Torch side using "AtenConvolutionOp".
    Value result = rewriter.create<AtenConvolutionOp>(loc, resultType, xTorchTensor,
        wTorchTensor, bTorchTensor, stridesList, padsList, dilationList,
        transposeVal, outputPadsList, groupTorchInt);
    setLayerNameAttr(op, result.getDefiningOp());
    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op.getResult().getType(), result);
    return success();
  }
};

void populateLoweringONNXToTorchConvOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConvOpToTorchLowering>(typeConverter, ctx);
}
