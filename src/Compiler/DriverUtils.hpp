/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DriverUtils.hpp - Utils for compiler driver  -------------===//
//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to create exec commands.
//
// This file should not include any ONNX-MLIR / MLIR / LLVM dependences except
// for onnx-mlir/include.
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DRIVER_UTILS_H
#define ONNX_MLIR_DRIVER_UTILS_H

#include <string>
#include <vector>

#include "include/onnx-mlir/Compiler/OMCompilerTypes.h"

namespace onnx_mlir {

std::vector<std::string> parseFlags(const std::string &flags);

// Append current working directory if filename is relative. When filename is
// empty, return empty path.
std::string getAbsolutePathUsingCurrentDir(const std::string &filename);

// Get the file name the input, first from inputFilename parameter, and if
// empty, scan the compiler flags to determine an input file name from there
// (aka a non -xxx option whose name ends in .onnx, .onnxtext, or .mlir).
std::string getInputFilename(
    const std::string &inputFilename, const std::vector<std::string> &flags);

// Get the name of the output, first from scanning the compiler flags for a -o
// option, and if not found, from the basename of the input file name (using
// getInputFilename above).
std::string getOutputFilename(
    const std::string &inputFilename, const std::vector<std::string> &flagVect);

// Method that derives the proper output name for a given output base name and a
// given compiler target (e.g. -EmitJNI).
std::string getTargetFilename(
    const std::string &outputBasename, EmissionTargetType target);

// Get the model tag by scanning the compilation flags.
std::string getModelTag(const std::vector<std::string> &flagVect);

} // namespace onnx_mlir

#endif // ONNX_MLIR_DRIVER_UTILS_H
