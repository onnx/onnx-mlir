/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DriverUtils.hpp - Utils for compiler driver  -------------===//
//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to create exec commands.
//
// This file should not include any ONNX-MLIR / MLIR / LLVM dependences except
// for onnx-mlir/include.
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DRIVER_UTILS_H
#define ONNX_MLIR_DRIVER_UTILS_H

#include <string>
#include <vector>

#include "include/onnx-mlir/Compiler/OMCompilerTypes.h"

namespace onnx_mlir {

std::vector<std::string> parseFlags(const std::string &flags);

std::string getOutputFilename(
    const std::string &inputFileName, const std::vector<std::string> &flagVect);

std::string getTargetFilename(
    const std::string &outputBasename, EmissionTargetType target);

} // namespace onnx_mlir

#endif // ONNX_MLIR_DRIVER_UTILS_H
