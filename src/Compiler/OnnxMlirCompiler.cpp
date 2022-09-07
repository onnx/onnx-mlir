/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/OnnxMlirCompiler.h"
#include "ExternalUtil.hpp"
#include "src/Compiler/CompilerUtils.hpp"

using namespace mlir;
using namespace onnx_mlir;

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

#ifdef _WIN32
#define strtok_r strtok_s
#endif

ONNX_MLIR_EXPORT int64_t omCompileFromFileViaCommand(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char **outputFilename, const char *flags, const char **errorMessage) {
  // Manually process the flags
  // Save the result string vector after processing
  std::vector<std::string> flagsVector;
  // Use the same standard as std::isspace to define white space characters
  const char delimiters[6] = {0x20, 0x0c, 0x0a, 0x0d, 0x09, 0x0b};
  // Use strtok_r instead of strtok because strtok_r is thread safe
  char *token;
  char *buffer = new char[std::strlen(flags) + 1];
  std::strcpy(buffer, flags);
  char *rest = buffer;
  while ((token = strtok_r(rest, delimiters, &rest)) != NULL) {
    flagsVector.push_back(std::string(token));
  }
  // Use 'onnx-mlir' command to compile the model.
  std::string onnxmlirPath = getToolPath("onnx-mlir");
  struct Command onnxmlirCompile(
      /*exePath=*/!onnxmlirPath.empty() ? onnxmlirPath : kOnnxmlirPath);

  for (std::size_t i = 0; i < flagsVector.size(); i++)
    onnxmlirCompile.appendStr(flagsVector[i]);

  // Indicate output name.
  std::string outputBaseNameStr(outputBaseName);
  onnxmlirCompile.appendStr("-o");
  onnxmlirCompile.appendStr(outputBaseNameStr);
  // Indicate input name.
  onnxmlirCompile.appendStr(std::string(inputFilename));
  // Run command.
  int rc = onnxmlirCompile.exec();
  if (rc == CompilerSuccess && outputFilename) {
    // Copy Filename
    std::string name = getTargetFilename(outputBaseNameStr, emissionTarget);
    *outputFilename = strdup(name.c_str());
  }
  return rc != 0 ? CompilerFailureInLLVMOpt : CompilerSuccess;
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

  std::string outputBaseNameStr(outputBaseName);
  rc = compileModule(module, context, outputBaseNameStr, emissionTarget);
  if (rc == CompilerSuccess && outputFilename) {
    // Copy Filename
    std::string name = getTargetFilename(outputBaseNameStr, emissionTarget);
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

  std::string outputBaseNameStr(outputBaseName);
  rc = compileModule(module, context, outputBaseNameStr, emissionTarget);
  if (rc == CompilerSuccess && outputFilename) {
    // Copy Filename
    std::string name = getTargetFilename(outputBaseNameStr, emissionTarget);
    *outputFilename = strdup(name.c_str());
  }
  return rc;
}

} // namespace onnx_mlir
}
