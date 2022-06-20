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
// Type conversion from ONNX to torch types.
//===-----------------------------------------------------------------===//

TorchTypeConverter::TorchTypeConverter() {
  // The order of type conversion is important: later ones are tried earlier.
  addConversion([](Type type) { return type; });

  addConversion([](StringType stringType) {
    return Torch::StringType::get(stringType.getContext());
  });

   // Use UnrealizedConversionCast as the bridge so that we don't need to pull in
   // patterns for other dialects.
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
