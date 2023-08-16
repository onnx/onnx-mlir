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
    std::string inputFileName, std::string flags, bool defaultEntryPoint,
    bool reuseCompiledModel)
    : onnx_mlir::PyExecutionSessionBase() /* constructor without Init */ {
  // First compile the onnx file.
  this->inputFileName = inputFileName;
  if (this->inputFileName.empty())
    throw std::runtime_error(reportLibraryOpeningError(inputFileName));

  char *outputName = nullptr;
  char *errorMsg = nullptr;
  if (reuseCompiledModel) {
    // see if there is a model to reuse.
    outputName = omCompileOutputFileName(inputFileName.c_str(), flags.c_str());
    FILE *file = fopen(outputName, "r");
    if (file)
      // File exists, we are ok.
      fclose(file);
    else
      // File does not exist, cannot reuse compilation.
      reuseCompiledModel = false;
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
      free(outputName);
      free(errorMsg);
      throw std::runtime_error(reportCompilerError(errorMessage));
    }
  }
  // Compilation success: save output file name.
  this->outputFileName = std::string(outputName);
  errorMessage = std::string();
  // Get the model tag from the compile flags.
  char *modelTag = omCompileModelTag(flags.c_str());
  // Now that we have a .so, initialize execution session.
  Init(this->outputFileName, modelTag, defaultEntryPoint);
  free(outputName);
  free(modelTag);
  free(errorMsg);
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
