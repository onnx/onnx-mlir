/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMCompileSession.cpp - compiler driver  ------------------===//
//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to compile onnx files using onnx-mlir.
//
// This file should not include any ONNX-MLIR / MLIR / LLVM dependences except
// for onnx-mlir/include.
//===----------------------------------------------------------------------===//

#include "src/Compiler/OMCompileSession.hpp"

#include <cstdlib>
#include <filesystem>

#include "src/Compiler/Command.hpp"
#include "src/Compiler/DriverUtils.hpp"
#include <onnx-mlir/Compiler/OMCompilerTypes.h>

using namespace onnx_mlir;
namespace fs = std::filesystem;

namespace onnx_mlir {

CompilerSession::CompilerSession() : successfullyCompiled(false) {}

CompilerSession::CompilerSession(const std::string &modelPath,
    const std::string &flags, const std::string &logFilename) {
  compile(modelPath, flags);
}

void CompilerSession::compile(const std::string &modelPath,
    const std::string &flags, const std::string &logFilename) {
  // Initialize state.
  successfullyCompiled = false;
  flagVect = parseFlags(flags);
  // When model path is given, add it to the vector of flags; otherwise locate
  // the input model filename from the flags.
  std::string inputFilename = onnx_mlir::getInputFilename(modelPath, flagVect);
  if (inputFilename.empty())
    throw CompilerSessionException(
        "Compilation failed: missing input model file");
  if (!fs::exists(inputFilename)) {
    throw CompilerSessionException(
        "Compilation failed: could not locate input model file \"" +
        inputFilename + "\"");
  }
  // Determine onnx-mlir executable path.
#ifdef _WIN32
  std::string onnxMlirPath = "onnx-mlir.exe";
#else
  std::string onnxMlirPath = "onnx-mlir";
#endif
  // Execute onnx-mlir command with arguments.
  onnx_mlir::Command compile(onnxMlirPath, /*verbose*/ true);
  compile.appendList(flagVect);
  if (!modelPath.empty())
    compile.appendStr(inputFilename);
  compile.print();
  if (!logFilename.empty())
    compile.redirectExecStreams(logFilename);
  int status = compile.exec();
  if (status != OnnxMlirCompilerErrorCodes::CompilerSuccess) {
    throw CompilerSessionException(
        "Compilation failed with error code " + std::to_string(status));
  }
  // Success, save filename of output, using an absolute path to increase
  // success of dlopen calls..
  std::string name = onnx_mlir::getOutputFilename(inputFilename, flagVect);
  outputFilename = getAbsolutePathUsingCurrentDir(name);
  successfullyCompiled = true;
}

std::string CompilerSession::getOutputFilename() {
  if (!successfullyCompiled) {
    throw CompilerSessionException(
        "Compiler session: has no successfully compiled model");
  }
  return outputFilename;
}

std::string CompilerSession::getModelTag() {
  if (!successfullyCompiled) {
    throw CompilerSessionException(
        "Compiler session: has no successfully compiled model");
  }
  return onnx_mlir::getModelTag(flagVect);
}

/* static */ std::string getInputFilename(
    const std::string &modelPath, const std::string &flags) {
  std::vector<std::string> flagVect = parseFlags(flags);
  std::string filename = onnx_mlir::getInputFilename(modelPath, flagVect);
  if (filename.empty())
    throw CompilerSessionException(
        "Compiler session: no model is provided for the compilation");
  return filename;
}

/* static */ std::string CompilerSession::getOutputFilename(
    const std::string &modelPath, const std::string &flags) {
  std::vector<std::string> flagVect = parseFlags(flags);
  // Success, save filename of output, using an absolute path to increase
  // success of dlopen calls..
  std::string name = onnx_mlir::getOutputFilename(modelPath, flagVect);
  return getAbsolutePathUsingCurrentDir(name);
}

/* static */ std::string CompilerSession::getModelTag(
    const std::string &flags) {
  std::vector<std::string> flagVect = parseFlags(flags);
  return onnx_mlir::getModelTag(flagVect);
}

} // namespace onnx_mlir
