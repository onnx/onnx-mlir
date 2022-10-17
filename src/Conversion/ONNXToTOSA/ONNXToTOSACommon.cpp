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

#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include <cassert>

//===-----------------------------------------------------------------===//
// Type conversion from ONNX types to Torch types.
//===-----------------------------------------------------------------===//

TosaTypeConverter::TosaTypeConverter() {
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
}
