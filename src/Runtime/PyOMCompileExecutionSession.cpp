/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- PyOMCompileExecutionSession.cpp - PyOMCompileExecutionSession
// Implementation -----===//
//
//
// =============================================================================
//
// This file contains implementations of PyOMCompileExecutionSession class,
// which helps python programs to compile and run binary model libraries.
//
//===----------------------------------------------------------------------===//

#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/onnx_pb.h"
SUPPRESS_WARNINGS_POP

#include "PyOMCompileExecutionSession.hpp"

namespace onnx_mlir {

PyOMCompileExecutionSession::PyOMCompileExecutionSession(
    std::string inputFileName, std::string sharedLibPath, std::string flags,
    bool defaultEntryPoint)
    : onnx_mlir::PyExecutionSessionBase(sharedLibPath, defaultEntryPoint) {
  this->inputFileName = inputFileName;
  if (this->inputFileName.empty()) {
    errorMessage = "No OMCompileExecuteSession was created with the input file "
                   "name specified.";
  }
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

int64_t PyOMCompileExecutionSession::pyGetCompiledResult() { return this->rc; }

std::string PyOMCompileExecutionSession::pyGetCompiledFileName() {
  return this->sharedLibPath;
}

std::string PyOMCompileExecutionSession::pyGetErrorMessage() {
  return this->errorMessage;
}

} // namespace onnx_mlir
