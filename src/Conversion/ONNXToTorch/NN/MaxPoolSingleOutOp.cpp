/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===- MaxPoolSingleOutOp.cpp - ONNX Op Transform -===//
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

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 * ONNX MaxPool operation
 *
 * ONNX MaxPool operation with a single output.
 * See ONNXMaxPoolOp for a full description of the MaxPool semantics.
 *
 * Attributes:
 *  auto_pad	::mlir::StringAttr	string attribute
 *  ceil_mode	::mlir::IntegerAttr	64-bit signed integer attribute
 *  dilations	::mlir::ArrayAttr	64-bit integer array attribute
 *  kernel_shape  ::mlir::ArrayAttr	64-bit integer array attribute
 *  pads	  ::mlir::ArrayAttr	64-bit integer array attribute
 *  storage_order ::mlir::IntegerAttr	64-bit signed integer attribute
 * strides	  ::mlir::ArrayAttr	64-bit integer array attribute
 *
 * Operands:
 * X	memref of any type values or tensor of any type values
 *
 * Results:
 * o_Y	memref of any type values or tensor of any type values
 *
 */

struct ONNXMaxPoolSingleOutOpToTorchLowering : public ConversionPattern {
public:
  ONNXMaxPoolSingleOutOpToTorchLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXMaxPoolSingleOutOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    ONNXMaxPoolSingleOutOp op1 = llvm::dyn_cast<ONNXMaxPoolSingleOutOp>(op);
    mlir::MLIRContext *context = op1.getContext();
    Location loc = op1.getLoc();

    Value x = op1.X();                               // ONNX operands
    auto autopad = op1.auto_padAttr();               // ::mlir::StringAttr
    auto dilations = op1.dilationsAttr();            // ::mlir::ArrayAttr
    auto kernalShape = op1.kernel_shapeAttr();       // ::mlir::ArrayAttr
    auto pads = op1.padsAttr();                      // ::mlir::ArrayAttr
    auto strides = op1.stridesAttr();                // ::mlir::ArrayAttr
    int64_t ceilingMode = op1.ceil_mode();           // int64_t
    auto ceilingModeAttr = op1.ceil_modeAttr();      // ::mlir::IntegerAttr
    auto storageOrderAttr = op1.storage_orderAttr(); // ::mlir::IntegerAttr
    int64_t storageOrder = op1.storage_order();      // int64_t

    // Reading the ONNX side pads values and store in the array
    auto intType = IntegerType::get(op1.getContext(), 64);
    auto boolType = IntegerType::get(op1.getContext(), 1);

    // Get mlir attributes as vectors
    std::vector<Value> translatePadsList =
        createPadsArrayAttribute(pads, intType, loc, rewriter);
    // Dilation has a default value of 1
    std::vector<Value> dilationOnnxList =
        createArrayAttribute(dilations, intType, loc, rewriter, 1);
    std::vector<Value> kernalShapeOnnxList;
    std::vector<Value> stridesOnnxList;

    if (kernalShape) {
      for (unsigned i = 0; i < kernalShape.size(); i++) {
        auto kernalShapeElement = IntegerAttr::get(intType,
            (kernalShape[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
        Value kernalShapeConstInt =
            rewriter.create<ConstantIntOp>(loc, kernalShapeElement);
        kernalShapeOnnxList.push_back(kernalShapeConstInt);
      }
    }

    if (strides) {
      for (unsigned i = 0; i < strides.size(); i++) {
        auto strideElement = IntegerAttr::get(intType,
            (strides[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
        Value strideElementConstInt =
            rewriter.create<ConstantIntOp>(loc, strideElement);
        stridesOnnxList.push_back(strideElementConstInt);
      }
    }

    // If ceilingMode is 0 (default) use floor rounding when computing the output shape, else use ceil.
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
        ValueRange{translatePadsList});
    Value dilationList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{dilationOnnxList});
    Value kernalShapeList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{kernalShapeOnnxList});

    // Determine input and result type
    TensorType inputTensorType = x.getType().cast<TensorType>();
    auto inputType = Torch::ValueTensorType::get(
        context, inputTensorType.getShape(), inputTensorType.getElementType());
    auto inputTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, inputType, x);

    TensorType opTensorType = op->getResult(0).getType().cast<TensorType>();
    auto resultType = Torch::ValueTensorType::get(op1.getContext(),
        opTensorType.getShape(), opTensorType.getElementType());

    // Allow symmetric padding and create additonal padding op to support
    // asymmetric padding in `torch-mlir`
    Value result;
    if (translatePadsList.size() == 2) {
      result = rewriter.create<AtenMaxPool2dOp>(loc, resultType,
          inputTensor, kernalShapeList, stridesList, padsList, dilationList,
          ceilingModeVal);
    } else {
      std::vector<int64_t> padShape = inputTensorType.getShape();
      for (unsigned i = 0; i < 2; i++) {
        auto startDim = (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
        auto endDim = (pads[i + 2].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
        padShape[i + 2] += (startDim + endDim);
      }
      auto padType = Torch::ValueTensorType::get(context, llvm::makeArrayRef(padShape),
          inputTensorType.getElementType());

      // Construct zero padding op since `torch` does not support asymmetric
      // padding for maxpool2d
      IntegerAttr zeroAttr = IntegerAttr::get(intType, 0);
      Value zeroPad = rewriter.create<ConstantIntOp>(loc, zeroAttr);
      Value padTensor = rewriter.create<AtenConstantPadNdOp>(loc, padType,
          inputTensor, padsList, zeroPad);

      Value padValue = rewriter.create<ConstantIntOp>(loc, zeroAttr);
      Value zeroPadsList = rewriter.create<PrimListConstructOp>(loc,
          Torch::ListType::get(rewriter.getType<Torch::IntType>()),
          ValueRange{padValue, padValue});

      result = rewriter.create<AtenMaxPool2dOp>(loc, resultType,
          padTensor, kernalShapeList, stridesList, zeroPadsList, dilationList,
          ceilingModeVal);
    }

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);
    return success();
  }
};

void populateLoweringONNXToTorchMaxPoolSingleOutOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMaxPoolSingleOutOpToTorchLowering>(typeConverter, ctx);
}
