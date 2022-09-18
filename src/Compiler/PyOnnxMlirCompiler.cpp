/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- PyOnnxMlirCompiler.hpp - PyOnnxMlirCompiler Implementation
//-----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of PyOnnxMlirCompiler class, which helps
// python programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "PyOnnxMlirCompiler.hpp"

namespace onnx_mlir {

int64_t PyOnnxMlirCompiler::pyCompileFromFile(std::string flags) {
  const char *outputName, *errorMsg;
  int64_t rc;
  rc = omCompileFromFile(
      inputFileName.c_str(), flags.c_str(), &outputName, &errorMsg);
  if (rc == 0) {
    // Compilation success: save output file name.
    outputFileName = std::string(outputName);
    // Empty error.
    errorMessage = std::string();
  } else {
    // Compilation failure: save error message.
    errorMessage = std::string(errorMsg);
    // Empty output file name.
    outputFileName = std::string();
  }
  return rc;
}

int64_t PyOnnxMlirCompiler::pyCompileFromArray(
    std::string outputBaseName, EmissionTargetType emissionTarget) {
  const char *outputName, *errorMsg;
  int64_t rc;
  rc = omCompileFromArray(inputBuffer, inputBufferSize, outputBaseName.c_str(),
      emissionTarget, &outputName, &errorMsg);
  if (rc == 0) {
    // Compilation success: save output file name.
    outputFileName = std::string(outputName);
    // Empty error.
    errorMessage = std::string();
  } else {
    // Compilation failure: save error message.
    errorMessage = std::string(errorMsg);
    // Empty output file name.
    outputFileName = std::string();
  }
  return rc;
}

std::string PyOnnxMlirCompiler::pyGetCompiledFileName() {
  return outputFileName;
}
std::string PyOnnxMlirCompiler::pyGetErrorMessage() { return errorMessage; }

} // namespace onnx_mlir