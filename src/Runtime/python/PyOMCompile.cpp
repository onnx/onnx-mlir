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

void PyOMCompile::compile(const std::string &modelPath,
    const std::string &flags, const std::string &compilerPath,
    const std::string &logFilename) {
  // Let compilation exceptions propagate naturally to Python without
  // printing to stderr. Python code can handle and display exceptions
  // as needed, avoiding duplicate error messages.
  OMcompile.compile(modelPath, flags, compilerPath, logFilename);
}

// =============================================================================
// Getters

std::string PyOMCompile::pyGetOutputFilename() {
  return OMcompile.getOutputFilename();
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
