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
    : onnx_mlir::PyExecutionSessionBase() /* constructor without Init */,
      compilerSession() /* constructor without compilation */ {
  // First compile the onnx file.
  if (inputFileName.empty())
    throw std::runtime_error(reportLibraryOpeningError(inputFileName));

  // See if we can reuse a compilation (no check on model or flag
  // equivalencies).
  if (reuseCompiledModel) {
    bool fileExist = false;
    std::string outputName =
        CompilerSession::getOutputFilename(inputFileName, flags);
    if (!outputName.empty()) {
      FILE *file = fopen(outputName.c_str(), "r");
      if (file) {
        // File exists, we are ok.
        fileExist = true;
        fclose(file);
      }
      // Reuse if file exists.
      reuseCompiledModel = fileExist;
    }
  }
  // Must compile?
  if (!reuseCompiledModel) {
    try {
      compilerSession.compile(inputFileName, flags);
    } catch (const onnx_mlir::CompilerSessionException &error) {
      errorMessage = "error during compiler session: ";
      errorMessage += error.what();
      std::cerr << errorMessage << std::endl;
      throw std::runtime_error(reportCompilerError(errorMessage));
    }
    // Now that we have a .so, initialize execution session.
    Init(compilerSession.getOutputFilename(), compilerSession.getModelTag(),
        defaultEntryPoint);
  }
}

// =============================================================================
// Custom getters

int64_t PyOMCompileExecutionSession::pyGetCompiledResult() {
  return compilerSession.success();
}

std::string PyOMCompileExecutionSession::pyGetCompiledFileName() {
  return compilerSession.getOutputFilename();
}

// hi alex: Discontinued, should use exception handling.
std::string PyOMCompileExecutionSession::pyGetErrorMessage() {
  return errorMessage;
}

} // namespace onnx_mlir
