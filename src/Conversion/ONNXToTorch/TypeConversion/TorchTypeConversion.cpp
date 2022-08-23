/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- TorchTypeConversion.cpp - ONNX types to Torch types conversion
//---------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// ======================================================================================
//
// This file contains code to setup type conversions from ONNX types (builtin)
// to Torch types (e.g. torch.tensor)
//
//===-------------------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;

void onnx_mlir::getTorchTypeConversionDependentDialects(
    DialectRegistry &registry) {
  // registry.insert<TorchConversionDialect>();
}

//===----------------------------------------------------------------------===//
// Type conversion setup.
//===----------------------------------------------------------------------===//

static torch::Torch::ValueTensorType getValueTensorFromBuiltinTensor(
    TensorType type) {
  auto context = type.getContext();
  if (type.isa<RankedTensorType>()) {
    return torch::Torch::ValueTensorType::get(
        context, type.getShape(), type.getElementType());
  }
  return torch::Torch::ValueTensorType::get(
      context, None, type.getElementType());
}

static void setupTensorToValueTensorConversion(
    ConversionTarget &target, TypeConverter &typeConverter) {
  target.addLegalOp<UnrealizedConversionCastOp>();
  typeConverter.addConversion([](TensorType type) -> Optional<Type> {
    return getValueTensorFromBuiltinTensor(type);
  });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, Torch::ValueTensorType type, ValueRange inputs,
          Location loc) -> Value {
        assert(inputs.size() == 1);
        assert(inputs[0].getType().isa<TensorType>());
        return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
            .getResult(0);
      });
  auto sourceMaterialization = [](OpBuilder &builder, TensorType type,
                                   ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::BaseTensorType>());
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
        .getResult(0);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

void onnx_mlir::setupTorchTypeConversion(
    ConversionTarget &target, TypeConverter &typeConverter) {
  setupTensorToValueTensorConversion(target, typeConverter);
}
