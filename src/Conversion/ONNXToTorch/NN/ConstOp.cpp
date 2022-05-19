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

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 * ONNX Constant  operation
 *
 * Creates the constant tensor.
 *
 * Operands : None.
 *
 * Validation
 * ----------
 * /scripts/docker/build_with_docker.py --external-build --build-dir build
 * --command
 * "build/Ubuntu1804-Release/third-party/onnx-mlir/Release/bin/onnx-mlir
 * --EmitONNXIR --debug --run-torch-pass
 * third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/node/
 * test_constant/model.onnx"
 *
 * Limitations
 * -----------
 * uses literal.
 *
 * TODO: Not handling String attribute in the ConstOp.
 */
class ONNXConstOpToTorchLowering : public ConversionPattern {
public:
  ONNXConstOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXConstantOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    mlir::MLIRContext *context = op->getContext();
    ONNXConstantOp op1 = llvm::dyn_cast<ONNXConstantOp>(op);

    auto valueAttribute = op1.valueAttr(); // ::mlir::Attribute

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
      if (auto intType = valueAttribute.getType().cast<IntegerType>()) {
        elementType = ::mlir::IntegerType::get(
            context, intType.getWidth(), IntegerType::Signed);
        valueAttrFinalized = valueAttribute;
      } else if (valueAttribute.getType().cast<::mlir::FloatType>()) {
        elementType = ::mlir::FloatType::getF32(context);
        valueAttrFinalized = valueAttribute;
      }
    }
    auto resultType = Torch::ValueTensorType::get(
        op1.getContext(), opTensorType.getShape(), elementType);

    Value result = rewriter.create<Torch::ValueTensorLiteralOp>(
        loc, resultType, valueAttrFinalized);

    llvm::outs() << "ValueTensorLiteralOp operation creation"
                 << "\n"
                 << result << "\n"
                 << "\n";
    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);
    return success();
  }
};

void populateLoweringONNXToTorchConstOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstOpToTorchLowering>(typeConverter, ctx);
}
