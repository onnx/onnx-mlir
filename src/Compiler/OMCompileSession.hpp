/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMCompilerSession.hpp - compiler driver  -----------------===//
//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to compile onnx files using onnx-mlir.
//
// This file should not include any ONNX-MLIR / MLIR / LLVM dependences except
// for onnx-mlir/include.
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_COMPILER_SESSION
#define ONNX_MLIR_COMPILER_SESSION

#include <string>
#include <vector>

// TODO: should ExecutionSession and CompilerSession be in the onnx_mlir
// namespace? They should not depend at all on the onnx-mlir compiler files
// (except implicitly).
namespace onnx_mlir {

// Exception class
class CompilerSessionException : public std::runtime_error {
public:
  explicit CompilerSessionException(const std::string &msg)
      : std::runtime_error(msg) {}
};

/*  C++ interface to compile an onnx model from a file via onnx-mlir command.
 *  This interface is thread safe, and does not take any flags from the
 *  current environment. All flags are passed by using the flags parameter,
 *  including the "-o output-file-name" option or the "-EmitXXX" options. All
 *  options that are available to onnx-mlir are also available here.
 *
 *  This call rely on executing onnx-mlir compiler.
 *
 *  When generating libraries or jar files, the compiler will link in
 *  lightweight runtimes / jar files. If these libraries / jar files are not in
 *  the system wide directory (typically /usr/local/lib), the user can override
 *  the default location using the ONNX_MLIR_LIBRARY_PATH environment variable.
 *
 *  @param inputFilename File name pointing onnx model protobuf or MLIR.
 *  Name may include a path, and must include the file name and its extention.
 *  If left empty, then the flags are expected to include the input model name.
 *
 *  @param flags A string that contains all the options provided to compile the
 *  model.
 *
 *  Trow CompilerSessionException on compiler error.
 */

class CompilerSession {
public:
  // Default constructor (compilation deferred to invocation of this->compile).
  CompilerSession();
  // Constructor that compiles model. Trow CompilerSessionException on compiler
  // error.
  CompilerSession(const std::string &modelPath, const std::string &flags,
      const std::string &logFilename = {});
  ~CompilerSession() = default;

  // Compile. Trow CompilerSessionException on compiler error.
  void compile(const std::string &modelPath, const std::string &flags,
      const std::string &logFilename = {});

  // File name of compiler generated model as compiled. Throw error if called
  // before a successfully compiled model.
  std::string getOutputFilename();

  // Model tag for the compiler generated model as compiled by the constructor.
  // Throw error if called before a successfully compiled model.
  std::string getModelTag();

  bool isSuccessfullyCompiled() { return successfullyCompiled; }

  // Functions to support caching, where we may want to know the output file
  // name and/or tag before compiling.
  static std::string getInputFilename(
      const std::string &modelPath, const std::string &flags);
  static std::string getOutputFilename(
      const std::string &modelPath, const std::string &flags);
  static std::string getModelTag(const std::string &flags);

private:
  std::vector<std::string> flagVect;
  std::string outputFilename;
  bool successfullyCompiled;
};

} // namespace onnx_mlir

#endif // ONNX_MLIR_COMPILER_SESSION
