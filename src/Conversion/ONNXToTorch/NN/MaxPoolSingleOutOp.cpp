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
 * “ONNX MaxPool operation with a single output.
 * ” “See ONNXMaxPoolOp for a full description of the MaxPool semantics.”
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

    // Reading the ONNX side pads values and store in the array.

    auto intType = IntegerType::get(op1.getContext(), 64);
    auto boolType = IntegerType::get(op1.getContext(), 1);

    std::vector<Value> translatePadsList =
        createPadsArrayAttribute(pads, intType, loc, rewriter);
    // reading the dilation values.
    std::vector<Value> dilationOnnxList =
        createArrayAttribute(dilations, intType, loc, rewriter, 1);
    std::vector<Value> kernalShapeOnnxList;
    std::vector<Value> stridesOnnxList;

    // reading the kernalShape values.
    if (kernalShape) {
      for (unsigned int i = 0; i < kernalShape.size(); i++) {
        auto kernalShapeElement = IntegerAttr::get(intType,
            (kernalShape[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
        Value kernalShapeConstInt =
            rewriter.create<ConstantIntOp>(loc, kernalShapeElement);
        kernalShapeOnnxList.push_back(kernalShapeConstInt);
      }
    }

    // reading the strides values.
    if (strides) {
      for (unsigned int i = 0; i < strides.size(); i++) {
        auto strideElement = IntegerAttr::get(intType,
            (strides[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
        Value strideElementConstInt =
            rewriter.create<ConstantIntOp>(loc, strideElement);
        stridesOnnxList.push_back(strideElementConstInt);
      }
    }

    // reading the ceilingMode values.
    // if ceilingMode is 0 means it's false, else true.
    Value constBoolOpValue = rewriter.create<ConstantBoolOp>(loc, false);
    Value ceilingModeVal;
    if (ceilingModeAttr) {
      if (ceilingMode == 0)
        ceilingModeVal = rewriter.create<ConstantBoolOp>(loc, false);
      else
        ceilingModeVal = rewriter.create<ConstantBoolOp>(loc, true);
    } else
      ceilingModeVal = constBoolOpValue;

    Value stridesList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{stridesOnnxList});

    Value dilationList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{dilationOnnxList});

    Value padsList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{translatePadsList});

    Value kernalShapeList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{kernalShapeOnnxList});

    TensorType xTensorType = x.getType().cast<TensorType>();
    TensorType opTensorType = op->getResult(0).getType().cast<TensorType>();

    auto xType = Torch::ValueTensorType::get(
        context, xTensorType.getShape(), xTensorType.getElementType());
    auto resultType = Torch::ValueTensorType::get(op1.getContext(),
        opTensorType.getShape(), opTensorType.getElementType());
    auto xTorchTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, xType, x);

    Value result = rewriter.create<AtenMaxPool2dOp>(loc, resultType,
        xTorchTensor, kernalShapeList, stridesList, padsList, dilationList,
        ceilingModeVal);
    llvm::outs() << "AtenMaxPool2dOp operation creation"
                 << "\n"
                 << result << "\n"
                 << "\n";
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
