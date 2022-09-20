/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- PyCompileExecutionSession.cpp - PyCompileExecutionSession
// Implementation -----===//
//
//
// =============================================================================
//
// This file contains implementations of PyCompileExecutionSession class, which
// helps python programs to compile and run binary model libraries.
//
//===----------------------------------------------------------------------===//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "PyCompileExecutionSession.hpp"

namespace onnx_mlir {

PyCompileExecutionSession::PyCompileExecutionSession(std::string inputFileName,
    std::string sharedLibPath, std::string flags, bool defaultEntryPoint)
    : onnx_mlir::PyExecutionSession(sharedLibPath, defaultEntryPoint) {
  this->inputFileName = inputFileName;
  const char *outputName, *errorMsg;
  int64_t rc;
  rc = omCompileFromFile(
      inputFileName.c_str(), flags.c_str(), &outputName, &errorMsg);
  if (rc == 0) {
    // Compilation success: save output file name.
    this->sharedLibPath = std::string(outputName);
    // Empty error.
    errorMessage = std::string();
  } else {
    // Compilation failure: save error message.
    errorMessage = std::string(errorMsg);
    // Empty output file name.
    this->sharedLibPath = std::string();
  }
}

int64_t PyCompileExecutionSession::pyGetCompiledResult() { return this->rc; }

std::string PyCompileExecutionSession::pyGetCompiledFileName() {
  return this->sharedLibPath;
}

std::string PyCompileExecutionSession::pyGetErrorMessage() {
  return this->errorMessage;
}

} // namespace onnx_mlir