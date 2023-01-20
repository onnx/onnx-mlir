/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- TorchTypeConversion.hpp - ONNX types to Torch types conversion
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

#ifndef ONNXMLIR_DIALECT_TORCHTYPECONVERSION_H
#define ONNXMLIR_DIALECT_TORCHTYPECONVERSION_H

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace onnx_mlir {

/// Get the dependent dialects which might be involved in a backend type
/// conversion.
void getTorchTypeConversionDependentDialects(DialectRegistry &registry);

void setupTorchTypeConversion(
    ConversionTarget &target, TypeConverter &typeConverter);
} // namespace onnx_mlir

#endif // ONNXMLIR_DIALECT_TORCHTYPECONVERSION_H
