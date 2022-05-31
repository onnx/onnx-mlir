/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnnxMlirCompiler.h"
#include "CompilerUtils.hpp"

extern "C" {
namespace onnx_mlir {

ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromEnv(const char *envVarName) {
  // ParseCommandLineOptions needs at least one argument
  std::string nameStr =
      "onnx-mlir (options from env var \"" + std::string(envVarName) + "\")";
  const char *argv[1];
  argv[0] = nameStr.c_str();
  const char *name = envVarName ? envVarName : OnnxMlirEnvOptionName.c_str();
  bool success = llvm::cl::ParseCommandLineOptions(
      1, argv, "SetCompilerOptionsFromEnv\n", nullptr, name);
  return success ? CompilerSuccess : InvalidCompilerOption;
}

ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromArgs(
    int64_t argc, char *argv[]) {
  bool success = llvm::cl::ParseCommandLineOptions(
      argc, argv, "SetCompilerOptionsFromArgs\n");
  return success ? CompilerSuccess : InvalidCompilerOption;
}

ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromArgsAndEnv(
    int64_t argc, char *argv[], const char *envVarName) {
  const char *name = envVarName ? envVarName : OnnxMlirEnvOptionName.c_str();
  bool success = llvm::cl::ParseCommandLineOptions(
      argc, argv, "SetCompilerOptionsFromArgsAndEnv\n", nullptr, name);
  return success ? CompilerSuccess : InvalidCompilerOption;
}

ONNX_MLIR_EXPORT int64_t omSetCompilerOption(
    const OptionKind kind, const char *val) {
  return setCompilerOption(kind, std::string(val));
}

ONNX_MLIR_EXPORT void omClearCompilerOption(const OptionKind kind) {
  clearCompilerOption(kind);
}

ONNX_MLIR_EXPORT const char *omGetCompilerOption(const OptionKind kind) {
  std::string val = getCompilerOption(kind);
  return strdup(val.c_str());
}

ONNX_MLIR_EXPORT int64_t omCompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char **outputFilename, const char **errorMessage) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::MLIRContext context;
  registerDialects(context);

  std::string internalErrorMessage;
  int rc = processInputFile(
      std::string(inputFilename), context, module, &internalErrorMessage);
  if (rc != CompilerSuccess) {
    if (errorMessage != NULL)
      *errorMessage = strdup(internalErrorMessage.c_str());
    return rc;
  }
  rc = compileModule(module, context, outputBaseName, emissionTarget);
  if (rc == CompilerSuccess && outputFilename) {
    // Copy Filename
    std::string name = getTargetFilename(outputBaseName, emissionTarget);
    *outputFilename = strdup(name.c_str());
  }
  return rc;
}

ONNX_MLIR_EXPORT int64_t omCompileFromArray(const void *inputBuffer,
    int bufferSize, const char *outputBaseName,
    EmissionTargetType emissionTarget, const char **outputFilename,
    const char **errorMessage) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::MLIRContext context;
  registerDialects(context);

  std::string internalErrorMessage;
  int rc = processInputArray(
      inputBuffer, bufferSize, context, module, &internalErrorMessage);
  if (rc != CompilerSuccess) {
    if (errorMessage != NULL)
      *errorMessage = strdup(internalErrorMessage.c_str());
    return rc;
  }
  rc = compileModule(module, context, outputBaseName, emissionTarget);
  if (rc == CompilerSuccess && outputFilename) {
    // Copy Filename
    std::string name = getTargetFilename(outputBaseName, emissionTarget);
    *outputFilename = strdup(name.c_str());
  }
  return rc;
}

} // namespace onnx_mlir
}
