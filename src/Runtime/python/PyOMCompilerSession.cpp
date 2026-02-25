/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- PyOMCompilerSession.cpp - PyOMCompilerSession Implementation ----===//
//
//
// =============================================================================
//
// This file contains implementations of PyOMCompilerSession class,
// which helps python programs to compile and run binary model libraries.
//
//===----------------------------------------------------------------------===//

#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/onnx_pb.h"
SUPPRESS_WARNINGS_POP

#include "PyOMCompilerSession.hpp"

namespace onnx_mlir {

// =============================================================================
// Constructor

PyOMCompilerSession::PyOMCompilerSession(std::string modelPath,
    std::string flags, const std::string &logFilename, bool reuseCompiledModel)
    : compilerSession() /* constructor without compilation */,
      modelPath(modelPath), flags(flags) {
  // First compile the onnx file.
  if (modelPath.empty())
    throw std::runtime_error("OMCompileSession: no input model provided");

  // See if we can reuse a compilation (no check on model or flag
  // equivalencies).
  if (reuseCompiledModel) {
    reuseCompiledModel = false; // Assume failure unless otherwise proven.
    std::string filename = CompilerSession::getOutputFilename(modelPath, flags);
    if (!filename.empty()) {
      FILE *file = fopen(filename.c_str(), "r");
      if (file) {
        // File exists, save
        reuseCompiledModel = true;
        fclose(file);
      }
    }
  }
  // Must compile?
  if (!reuseCompiledModel) {
    try {
      compilerSession.compile(modelPath, flags, logFilename);
    } catch (const onnx_mlir::CompilerSessionException &error) {
      std::string errorMessage =
          "OMCompileSession: compilation failed with error ";
      errorMessage += error.what();
      std::cerr << errorMessage << std::endl;
      throw std::runtime_error(errorMessage);
    }
  }
}

// =============================================================================
// Custom getters

std::string PyOMCompilerSession::pyGetOutputFilename() {
  return onnx_mlir::CompilerSession::getOutputFilename(modelPath, flags);
}

std::string PyOMCompilerSession::pyGetModelTag() {
  return onnx_mlir::CompilerSession::getModelTag(flags);
}

} // namespace onnx_mlir
