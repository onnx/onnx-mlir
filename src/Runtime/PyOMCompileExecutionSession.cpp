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

// =============================================================================
// Constructor

PyOMCompileExecutionSession::PyOMCompileExecutionSession(
    std::string inputFileName, std::string flags,
    bool defaultEntryPoint, bool reuseCompiledModel)
    : onnx_mlir::PyExecutionSessionBase() /* constructor without Init */ {
  // First compile the onnx file.
  this->inputFileName = inputFileName;
  if (this->inputFileName.empty())
    throw std::runtime_error(reportLibraryOpeningError(inputFileName));

  const char *outputName, *errorMsg;
  if (reuseCompiledModel) {
    // see if there is a model to reuse.
    outputName = omCompileOutputFileName(inputFileName.c_str(), flags.c_str());
    bool fileExists = access(outputName, F_OK) != -1;
    if (!fileExists) {
      fprintf(stderr, "file `%s' does not exists, compile.\n", outputName);
      reuseCompiledModel = false;
    }
  }
  if (!reuseCompiledModel) {
    int64_t rc;
    rc = omCompileFromFile(
        inputFileName.c_str(), flags.c_str(), &outputName, &errorMsg);
    if (rc != 0) {
      // Compilation failure: save error message.
      errorMessage = std::string(errorMsg);
      // Empty output file name.
      this->outputFileName = std::string();
      throw std::runtime_error(reportCompilerError(errorMessage));
    }
  }
  // Compilation success: save output file name.
  this->outputFileName = std::string(outputName);
  errorMessage = std::string();
  // Now that we have a .so, initialize execution session.
  Init(this->outputFileName, defaultEntryPoint);
}

// =============================================================================
// Custom getters

int64_t PyOMCompileExecutionSession::pyGetCompiledResult() { return rc; }

std::string PyOMCompileExecutionSession::pyGetCompiledFileName() {
  return outputFileName;
}

std::string PyOMCompileExecutionSession::pyGetErrorMessage() {
  return errorMessage;
}

} // namespace onnx_mlir
