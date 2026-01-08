//===- ONNXBufferizableOpInterface.hpp - ONNX Bufferizable Interface ------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the BufferizableOpInterface registration for ONNX
// operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/DialectRegistry.h"

namespace onnx_mlir {

// Register BufferizableOpInterface for ONNX operations.
// This allows One-Shot Bufferization to work with mixed Linalg and ONNX
// operations in the same IR.
void registerONNXBufferizableOpInterfaces(mlir::DialectRegistry &registry);

} // namespace onnx_mlir

