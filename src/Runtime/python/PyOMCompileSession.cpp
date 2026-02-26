/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- PyOMCompileSession.cpp - PyOMCompileSession Implementation ------===//
//
// Copyright 2024-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of PyOMCompileSession class,
// which helps python programs to compile and run binary model libraries.
//
//===----------------------------------------------------------------------===//

#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/onnx_pb.h"
SUPPRESS_WARNINGS_POP

#include "PyOMCompileSession.hpp"

namespace onnx_mlir {

// =============================================================================
// Constructor

PyOMCompileSession::PyOMCompileSession(std::string modelPath, std::string flags,
    const std::string &logFilename, bool reuseCompiledModel)
    : compilerSession() /* constructor without compilation */,
      modelPath(modelPath), flags(flags) {

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
      std::string errorMessage = error.what();
      std::cerr << errorMessage << std::endl;
      throw std::runtime_error(errorMessage);
    }
  }
}

// =============================================================================
// Custom getters

std::string PyOMCompileSession::pyGetOutputFilename() {
  return onnx_mlir::CompilerSession::getOutputFilename(modelPath, flags);
}

std::string PyOMCompileSession::pyGetModelTag() {
  return onnx_mlir::CompilerSession::getModelTag(flags);
}

} // namespace onnx_mlir
