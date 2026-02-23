/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMCompilerDriver.hpp - compiler driver  -----------------===//
//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to compile onnx files using onnx-mlir.
//
// This file should not include any ONNX-MLIR / MLIR / LLVM dependences except
// for onnx-mlir/include.
//===----------------------------------------------------------------------===//

#ifndef COMPILER_DRIVER_H
#define COMPILER_DRIVER_H

#include <string>
#include <vector>

namespace onnx_mlir_compiler_driver {

extern int64_t omCompile(const std::string &inputFilename,
    const std::string &flags, std::string &outputFilename,
    std::string &errorMessage);

extern int64_t omCompileOuputFilename(const std::string &inputFilename,
    const std::string &flags, std::string &outputFilename);

extern int64_t omCompileModelTag(const std::string &flags, std::string tag);

} // namespace onnx_mlir_compiler_driver

#endif // COMPILER_DRIVER_H
