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
// Type conversion from Onnx types to Torch types.
//===-----------------------------------------------------------------===//

TorchTypeConverter::TorchTypeConverter() {
  // The order of type conversion is important: later ones are tried earlier.
  addConversion([](Type type) { return type; });

  addConversion([](StringType stringType) {
    return Torch::StringType::get(stringType.getContext());
  });

  addConversion([](TensorType tensorType) {
    assert(tensorType.hasRank() && "expected only ranked shapes");
    if (tensorType.getElementType().isa<StringType>()) {
      Type elementType = Torch::StringType::get(tensorType.getContext());
      return MemRefType::get(tensorType.getShape(), elementType);
    }
    return MemRefType::get(tensorType.getShape(), 
		    tensorType.getElementType());
  });

  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return llvm::None;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return llvm::None;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}
