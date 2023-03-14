/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ConstOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace mlir::torch;

// ONNX Constant  operation
//
// Creates the constant tensor.
class ONNXConstOpToTorchLowering : public OpConversionPattern<ONNXConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final  {

    Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    Attribute valueAttribute = op.getValueAttr(); // ::mlir::Attribute

    // Steps
    // 1) Extract float attributes array from ONNX and compare with
    //      the Netron file,
    // 2) Find the shape of this array in step 1,
    // 3) Create the result type,
    // 4) Create the torch tensor of shape as in 2,
    // 5) Create the torch op and replace it.

    TensorType opTensorType = op->getResult(0).getType().cast<TensorType>();
    ::mlir::Attribute valueAttrFinalized;
    Type elementType;
    if (opTensorType) {
      // ElementType is integer type.
      if (auto integerType =
              opTensorType.getElementType().dyn_cast<IntegerType>()) {
        elementType = IntegerType::get(
            context, integerType.getWidth(), IntegerType::Signed);
        // creating the Dense Attribute for the valueAttribute.
        auto denseValueAttr =
            valueAttribute.dyn_cast<::mlir::DenseElementsAttr>();
        // getting the shape  of the opTensorType
        ShapedType denseValueType =
            RankedTensorType::get(opTensorType.getShape(), elementType);
        std::vector<APInt> intValues;
        for (auto n : denseValueAttr.getValues<APInt>())
          intValues.push_back(n);
        auto newDenseValueAttr =
            DenseElementsAttr::get(denseValueType, intValues);
        valueAttrFinalized = newDenseValueAttr;
      } else if (auto floatType = opTensorType.getElementType()
                                      .dyn_cast<::mlir::FloatType>()) {
        // ElementType is float type
        elementType = ::mlir::FloatType::getF32(context);
        auto denseValueAttr =
            valueAttribute.dyn_cast<::mlir::DenseElementsAttr>();
        ShapedType denseValueType =
            RankedTensorType::get(opTensorType.getShape(), elementType);
        std::vector<APFloat> floatValues;
        for (auto n : denseValueAttr.getValues<APFloat>())
          floatValues.push_back(n);
        auto newDenseValueAttr =
            DenseElementsAttr::get(denseValueType, floatValues);
        valueAttrFinalized = newDenseValueAttr;
      } else {
        elementType = opTensorType.getElementType();
        valueAttrFinalized = valueAttribute;
      }
    } else {
      if (auto intType = valueAttribute.cast<TypedAttr>().getType().cast<IntegerType>()) {
        elementType = ::mlir::IntegerType::get(
            context, intType.getWidth(), IntegerType::Signed);
        valueAttrFinalized = valueAttribute;
      } else if (valueAttribute.cast<TypedAttr>().getType().cast<::mlir::FloatType>()) {
        elementType = ::mlir::FloatType::getF32(context);
        valueAttrFinalized = valueAttribute;
      }
    }
    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    Value result = rewriter.create<Torch::ValueTensorLiteralOp>(
        loc, resultType, valueAttrFinalized);

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, resultType, result);
    return success();
  }
};

void populateLoweringONNXToTorchConstOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstOpToTorchLowering>(typeConverter, ctx);
}
