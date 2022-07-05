/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====- ONNXToTorchCommon.cpp - ONNX dialects to Torch lowering -===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// ========================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the Torch dialect.
//
//===-----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

//===-----------------------------------------------------------------===//
// Type conversion from ONNX types to Torch types.
//===-----------------------------------------------------------------===//

TorchTypeConverter::TorchTypeConverter() {
  /// The order of type conversion is important: later ones are tried earlier.
  /// Legalize all remaining types to ensure successful conversions.
  addConversion([](Type type) { return type; });

  /// Unrealized conversion cast
  auto addUnrealizedCast = [&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return llvm::None;
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  };
  addSourceMaterialization(addUnrealizedCast);
  addTargetMaterialization(addUnrealizedCast);

  /// String conversion
  addConversion([](StringType stringType) {
    return Torch::StringType::get(stringType.getContext());
  });

  /// Torch tensor conversion
  addConversion([](TensorType type) -> Optional<Type> {
    return Torch::ValueTensorType::get(
      type.getContext(), type.getShape(), type.getElementType());
  });
  auto addTensorCast = [](OpBuilder &builder, TensorType type,
                          ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::BaseTensorType>());
    return builder.create<torch::TorchConversion::ToBuiltinTensorOp>(loc, inputs[0]);
  };
  addSourceMaterialization(addTensorCast);
  addArgumentMaterialization(addTensorCast);
  addTargetMaterialization([](OpBuilder &builder, Torch::ValueTensorType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<torch::TorchConversion::FromBuiltinTensorOp>(
      loc, type, inputs[0]);
  });

  /// Create tensor to value tensor conversion and ensure that we always
  /// use signed integer types. This is important since the `torch-mlir`
  /// only supports signed integer types.
  addConversion([](RankedTensorType type) -> Type {
    mlir::Type elementType = type.cast<TensorType>().getElementType();
    if (type.getElementType().isSignlessInteger()) {
      elementType = IntegerType::get(type.getContext(),
        type.getElementType().getIntOrFloatBitWidth(), IntegerType::Signed);
    }
    return Torch::ValueTensorType::get(type.getContext(), type.getShape(), elementType);
  });
  addTargetMaterialization([](OpBuilder &builder,
                              RankedTensorType type, ValueRange inputs,
                              Location loc) -> Optional<Value> {
    if (type.getElementType().isSignlessInteger()) {
      mlir::Type elementType = IntegerType::get(type.getContext(),
        type.getElementType().getIntOrFloatBitWidth(), IntegerType::Unsigned);
      return builder.create<UnrealizedConversionCastOp>(loc, elementType, inputs).getResult(0);
    }
    return llvm::None;
  });
}
