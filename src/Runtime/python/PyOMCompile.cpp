/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- PyOMCompile.cpp - PyOMCompile Implementation ------===//
//
// Copyright 2024-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of PyOMCompile class,
// which helps python programs to compile and run binary model libraries.
//
//===----------------------------------------------------------------------===//

#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/onnx_pb.h"
SUPPRESS_WARNINGS_POP

#include "PyOMCompile.hpp"

namespace onnx_mlir {

// =============================================================================
// Constructor

PyOMCompile::PyOMCompile(std::string modelPath, std::string flags,
    const std::string &logFilename, bool reuseCompiledModel)
    : OMcompile() /* constructor without compilation */, modelPath(modelPath),
      flags(flags) {

  // See if we can reuse a compilation (no check on model or flag
  // equivalencies).
  if (reuseCompiledModel) {
    reuseCompiledModel = false; // Assume failure unless otherwise proven.
    std::string filename = OMCompile::getOutputFilename(modelPath, flags);
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
    // Let compilation exceptions propagate naturally to Python without
    // printing to stderr. Python code can handle and display exceptions
    // as needed, avoiding duplicate error messages.
    // Old version caught and re-threw with stderr output, causing duplicates.
    OMcompile.compile(modelPath, flags, logFilename);
  }
}

// =============================================================================
// Custom getters

std::string PyOMCompile::pyGetOutputFilename() {
  return OMcompile.getOutputFilename();
}

std::string PyOMCompile::pyGetOutputConstantFilename() {
  return OMcompile.getOutputConstantFilename();
}

std::string PyOMCompile::pyGetModelTag() { return OMcompile.getModelTag(); }

} // namespace onnx_mlir
