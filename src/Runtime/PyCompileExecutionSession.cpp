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

PyCompileExecutionSession::PyCompileExecutionSession(
    std::string fileName, bool defaultEntryPoint)
    : onnx_mlir::PyExecutionSession(fileName, defaultEntryPoint) {
  inputFileName = fileName;
}

int64_t PyCompileExecutionSession::pyCompileFromFile(std::string flags) {
  const char *outputName, *errorMsg;
  int64_t rc;
  rc = omCompileFromFile(
      inputFileName.c_str(), flags.c_str(), &outputName, &errorMsg);
  if (rc == 0) {
    // Compilation success: save output file name.
    sharedLibPath = std::string(outputName);
    // Empty error.
    errorMessage = std::string();
  } else {
    // Compilation failure: save error message.
    errorMessage = std::string(errorMsg);
    // Empty output file name.
    sharedLibPath = std::string();
  }
  return rc;
}

std::string PyCompileExecutionSession::pyGetCompiledFileName() {
  return sharedLibPath;
}
std::string PyCompileExecutionSession::pyGetErrorMessage() {
  return errorMessage;
}

} // namespace onnx_mlir