/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- CompilerUtils.hpp -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "onnx-mlir/Compiler/OMCompilerTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

#include <optional>
#include <string>
#include <vector>

namespace onnx_mlir {

std::optional<std::string> getEnvVar(std::string name);

struct Command {

  std::string _path;
  std::vector<std::string> _args;

  Command(std::string exePath)
      : _path(std::move(exePath)),
        _args({llvm::sys::path::filename(_path).str()}) {}

  Command &appendStr(const std::string &arg);
  Command &appendStrOpt(const std::optional<std::string> &arg);
  Command &appendList(const std::vector<std::string> &args);
  Command &resetArgs();
  int exec(std::string wdir = "") const;
};

void loadDialects(mlir::MLIRContext &context);

// Get Tool path, see comments in CompilerUtils.cpp for more details.
std::string getToolPath(
    const std::string &tool, const std::string &systemToolPath);

// ProcessInput* return 0 on success, OnnxMlirCompilerErrorCodes on error.
int processInputFile(llvm::StringRef inputFilename, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *errorMessage);
int processInputArray(const void *onnxBuffer, int bufferSize,
    mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *errorMessage);

onnx_mlir::InputIRLevelType determineInputIRLevel(
    mlir::OwningOpRef<mlir::ModuleOp> &module);

// Returns 0 on success, OnnxMlirCompilerErrorCodes on failure.
int outputCode(mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string filenameWithExt, int64_t largeElementLimit = -1);

// Process the input model given by its module and context into an output file
// according to the emission target type. Name of the output file can be
// constructed using the getTargetFilename function below.  When  generating
// libraries or jar files, the compiler will link in lightweight runtimes / jar
// files. If these libraries / jar files are not in the system wide directory
// (typically /usr/local/lib), the user can override the default location using
// the ONNX_MLIR_LIBRARY_PATH environment variable.
// Returns 0 on success,OnnxMlirCompilerErrorCodes on failure.
int compileModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputNameNoExt,
    onnx_mlir::EmissionTargetType emissionTarget);

// Extend the input filename (with possibly a path but no extention) by the
// extention generated by the given emission target type. Names may be different
// depending on the underlying machine and/or operating system.
std::string getTargetFilename(
    const std::string filenameNoExt, EmissionTargetType target);
} // namespace onnx_mlir
