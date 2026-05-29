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

#include "PyOMCompile.hpp"

namespace onnx_mlir {

// =============================================================================
// Constructors

// Constructor for local compilation
PyOMCompile::PyOMCompile(const std::string &compilerPath, bool verbose)
    : OMcompile(compilerPath, verbose) {}

// Constructor for container-based compilation
PyOMCompile::PyOMCompile(const std::string &containerImage,
    const std::string &compilerPathInContainer, const std::string &engine,
    bool autoPull, bool verbose)
    : OMcompile(containerImage, compilerPathInContainer,
          engine == "docker"   ? OMCompile::ContainerEngine::Docker
          : engine == "podman" ? OMCompile::ContainerEngine::Podman
                               : OMCompile::ContainerEngine::Auto,
          autoPull, verbose) {}

// =============================================================================
// Compile method

std::string PyOMCompile::compile(const std::string &modelPath,
    const std::string &flags, const std::string &outputPath,
    const std::string &compilerPath, const std::string &logFilename,
    bool reuseCompiledModel) {
  std::string filename = pyPredictOutputFilename(modelPath, flags, outputPath);
  if (reuseCompiledModel) {
    if (!filename.empty()) {
      FILE *file = fopen(filename.c_str(), "r");
      if (file) {
        // File exists, no compilation is needed
        fclose(file);
        return filename;
      }
    }
  }

  // Let compilation exceptions propagate naturally to Python without
  // printing to stderr. Python code can handle and display exceptions
  // as needed, avoiding duplicate error messages.
  // Old version caught and re-threw with stderr output, causing duplicates.
  OMcompile.compile(modelPath, flags, outputPath, compilerPath, logFilename);
  // Check for compiler consistency
  assert(filename == OMcompile.getOutputFilename() &&
         "Something wrong with OMCompile.cpp");
  return OMcompile.getOutputFilename();
}

// =============================================================================
// Getters

std::string PyOMCompile::pyGetOutputFilename() {
  return OMcompile.getOutputFilename();
}

/* static */ std::string PyOMCompile::pyPredictOutputFilename(
    const std::string &modelPath, const std::string &flags,
    const std::string &outputPath) {
  return onnx_mlir::OMCompile::predictOutputFilename(
      modelPath, flags, outputPath);
}

std::string PyOMCompile::pyGetOutputConstantFilename() {
  return OMcompile.getOutputConstantFilename();
}

std::string PyOMCompile::pyGetModelTag() { return OMcompile.getModelTag(); }

bool PyOMCompile::pyIsSuccessfullyCompiled() {
  return OMcompile.isSuccessfullyCompiled();
}

bool PyOMCompile::pyHasOutputConstantFilename() {
  return OMcompile.hasOutputConstantFilename();
}

} // namespace onnx_mlir
