/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- PyOnnxMirCompiler.hpp - PyOnnxMirCompiler Implementation
//-----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of PyOnnxMirCompiler class, which helps
// python programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "PyOnnxMlirCompiler.hpp"

namespace onnx_mlir {

// Options
int64_t PyOnnxMirCompiler::pySetOptionsFromEnv(std::string envVarName) {
  return omSetCompilerOptionsFromEnv(envVarName.c_str());
}

int64_t PyOnnxMirCompiler::pySetOption(const OptionKind kind, std::string val) {
  return omSetCompilerOption(kind, val.c_str());
}

void PyOnnxMirCompiler::pyClearOption(const OptionKind kind) {
  omClearCompilerOption(kind);
}

std::string PyOnnxMirCompiler::pyGetOption(const OptionKind kind) {
  return std::string(omGetCompilerOption(kind));
}

int64_t PyOnnxMirCompiler::pyCompile(
    std::string outputBaseName, EmissionTargetType emissionTarget) {
  const char *outputName, *errorMsg;
  int64_t rc;
  if (inputBufferSize)
    rc = omCompileFromArray(inputBuffer, inputBufferSize,
        outputBaseName.c_str(), emissionTarget, &outputName, &errorMsg);
  else
    rc = omCompileFromFile(inputFileName.c_str(), outputBaseName.c_str(),
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

std::string PyOnnxMirCompiler::pyGetCompiledFileName() {
  return outputFileName;
}
std::string PyOnnxMirCompiler::pyGetErrorMessage() { return errorMessage; }

} // namespace onnx_mlir