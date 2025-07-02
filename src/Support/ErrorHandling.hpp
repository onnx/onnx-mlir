/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----------------------- ErrorHandling.hpp ---------------------------===//
//
// This file contains common error handling utilities for ONNX-MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ERROR_HANDLING_HPP
#define ONNX_MLIR_ERROR_HANDLING_HPP

#include "include/onnx-mlir/Compiler/OMCompilerTypes.h"

#include <system_error>

namespace std {
template <>
struct is_error_code_enum<onnx_mlir::OnnxMlirCompilerErrorCodes> : true_type {};
} // namespace std

namespace onnx_mlir {
[[nodiscard]] std::error_code make_error_code(OnnxMlirCompilerErrorCodes);
} // namespace onnx_mlir

#endif // ONNX_MLIR_ERROR_HANDLING_HPP
