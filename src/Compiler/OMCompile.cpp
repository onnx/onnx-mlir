/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMCompile.cpp - compiler driver  ------------------===//
//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to compile onnx model files in .onnx, .mlir, or
// .onnxtext using onnx-mlir.
//
// This file should not include any ONNX-MLIR / MLIR / LLVM dependences except
// for onnx-mlir/include.
//===----------------------------------------------------------------------===//

#include "src/Compiler/OMCompile.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>

#include "src/Compiler/Command.hpp"
#include "src/Compiler/CommandUtils.hpp"
#include <onnx-mlir/Compiler/OMCompilerTypes.h>

using namespace onnx_mlir;
namespace fs = std::filesystem;

namespace onnx_mlir {

void OMCompile::compile(const std::string &modelPath, const std::string &flags,
    const std::string &logFilename) {
  // Initialize state.
  successfullyCompiled = false;
  outputFilename = {};
  outputConstantFilename = {};
  flagVect = parseFlags(flags);
  bool verbose =
      std::find(flagVect.begin(), flagVect.end(), "-v") != flagVect.end();
  // When model path is given, add it to the vector of flags; otherwise locate
  // the input model filename from the flags.
  std::string inputFilename = onnx_mlir::getInputFilename(modelPath, flagVect);
  if (inputFilename.empty())
    throw OMCompileException("Compilation failed: missing input model file");
  if (!fs::exists(inputFilename)) {
    throw OMCompileException(
        "Compilation failed: could not locate input model file \"" +
        inputFilename + "\"");
  }
  // Determine the onnx-mlir executable path.
#ifdef _WIN32
  std::string compilerFilename = "onnx-mlir.exe";
#else
  std::string compilerFilename = "onnx-mlir";
#endif
  // Execute onnx-mlir command with arguments.
  int status;
  try {
    onnx_mlir::Command compileCommand(compilerFilename, verbose);
    compileCommand.appendList(flagVect);
    if (!modelPath.empty())
      compileCommand.appendStr(inputFilename);
    if (!logFilename.empty())
      compileCommand.redirectExecStreams(logFilename);
    // Compile, and catch CommandException if any.
    status = compileCommand.exec();
  } catch (const onnx_mlir::CommandException &error) {
    std::string errorMessage;
    if (verbose) {
      errorMessage = error.what();
      fprintf(stderr, "Return message from command exception: %s\n",
          errorMessage.c_str());
    }
    errorMessage =
        onnx_mlir::getOnnxMlirCompilerErrorDescription(CompilerCrashed);
    throw OMCompileException(
        "Compilation failed with error code: " + errorMessage);
  }
  // Compilation may have succeeded with an error code, catch here.
  if (status != OnnxMlirCompilerErrorCodes::CompilerSuccess) {
    std::string errorMessage(
        onnx_mlir::getOnnxMlirCompilerErrorDescription(status));
    throw OMCompileException(
        "Compilation failed with error code: " + errorMessage);
  }
  // Success, save filename of output, using an absolute path to increase
  // success of dlopen calls.. Below is why (according to man)
  /*
   * If you pass a path containing a slash (absolute or relative), e.g.
   * dlopen("./libfoo.so", RTLD_NOW), the loader uses that exact path (no
   * search). If you pass just a filename (no slash), the dynamic linker
   * searches in this order (simplified):
   *
   * 1. DT_RPATH of the calling object (if present and there is no DT_RUNPATH).
   *
   * 2. LD_LIBRARY_PATH (ignored for setuid/setgid for security).
   *
   * 3. DT_RUNPATH of the calling object (if present).
   *
   * 4. /etc/ld.so.cache entries (managed by ldconfig).
   *
   * 5. System defaults: /lib then /usr/lib. [man7.org]
   */
  std::string name = onnx_mlir::getOutputFilename(inputFilename, flagVect);
  outputFilename = getAbsolutePathUsingCurrentDir(name);
  successfullyCompiled = true;
  // Check if there is a output data filename.
  std::string constFilename = fs::path(outputFilename).stem().string();
  constFilename += ".constants.bin";
  if (fs::exists(constFilename))
    outputConstantFilename = constFilename;
}

std::string OMCompile::getOutputFilename() {
  if (!successfullyCompiled) {
    throw OMCompileException(
        "Compiler session: has no successfully compiled model");
  }
  return outputFilename;
}

std::string OMCompile::getOutputConstantFilename() {
  if (!successfullyCompiled) {
    throw OMCompileException(
        "Compiler session: has no successfully compiled model");
  }
  return outputConstantFilename;
}

std::string OMCompile::getModelTag() {
  if (!successfullyCompiled) {
    throw OMCompileException(
        "Compiler session: has no successfully compiled model");
  }
  return onnx_mlir::getModelTag(flagVect);
}

/* static */ std::string getInputFilename(
    const std::string &modelPath, const std::string &flags) {
  std::vector<std::string> flagVect = parseFlags(flags);
  std::string filename = onnx_mlir::getInputFilename(modelPath, flagVect);
  if (filename.empty())
    throw OMCompileException(
        "Compiler session: no model is provided for the compilation");
  return filename;
}

/* static */ std::string OMCompile::getOutputFilename(
    const std::string &modelPath, const std::string &flags) {
  std::vector<std::string> flagVect = parseFlags(flags);
  // Success, save filename of output, using an absolute path to increase
  // success of dlopen calls..
  std::string name = onnx_mlir::getOutputFilename(modelPath, flagVect);
  return getAbsolutePathUsingCurrentDir(name);
}

/* static */ std::string OMCompile::getModelTag(const std::string &flags) {
  std::vector<std::string> flagVect = parseFlags(flags);
  return onnx_mlir::getModelTag(flagVect);
}

} // namespace onnx_mlir
