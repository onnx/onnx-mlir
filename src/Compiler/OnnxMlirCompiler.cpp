/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- OnnxMlirCompiler.cpp - ONNX-MLIR Compiler API Declarations ---===//
//
// This file contains code for the onnx-mlir compiler functionality exported
// from the OnnxMlirCompiler library
//
//===----------------------------------------------------------------------===//

#include "include/OnnxMlirCompiler.h"
#include "ExternalUtil.hpp"
#include "src/Compiler/CompilerUtils.hpp"

using namespace mlir;
using namespace onnx_mlir;

extern "C" {
namespace onnx_mlir {

#if 0 // hi alex
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

#endif

static OnnxMlirCompilerErrorCodes pushErrorMessage(const char **errorMessage,
    const OnnxMlirCompilerErrorCodes error, const std::string &msg) {
  if (errorMessage)
    *errorMessage = strdup(msg.c_str());
  return error;
}

#ifdef _WIN32
#define strtok_r strtok_s
#endif

ONNX_MLIR_EXPORT int64_t omCompileFromFileViaCommand(const char *inputFilename,
    EmissionTargetType emissionTarget, const char *flags,
    const char **outputFilename, const char **errorMessage) {

  // Process the flags, saving each space-separated text in a separate
  // entry in the string vector flagsVector.
  std::vector<std::string> flagsVectorStr;
  // Use the same standard as std::isspace to define white space characters with
  // strtok_r instead of strtok to be thread safe.
  const char delimiters[6] = {0x20, 0x0c, 0x0a, 0x0d, 0x09, 0x0b};
  int len = std::strlen(flags);
  char *buffer = new char(len + 1);
  char *rest = buffer;
  char *token;
  std::strcpy(buffer, flags);
  while ((token = strtok_r(rest, delimiters, &rest)) != NULL) {
    flagsVectorStr.push_back(std::string(token));
  }
  delete buffer;

  // Use 'onnx-mlir' command to compile the model.
  std::string onnxMlirPath = getToolPath("onnx-mlir");
  struct Command onnxMlirCompile(
      /*exePath=*/!onnxMlirPath.empty() ? onnxMlirPath : kOnnxmlirPath);
  // Add each of the flags to the command, locating output base name if avail.
  std::string inputFilenameStr(inputFilename);
  std::string outputBasenameStr;
  std::size_t num = flagsVectorStr.size();
  for (std::size_t i = 0; i < num; ++i) {
    if (flagsVectorStr[i].find("-o") == 0) {
      if (i + 1 >= num)
        return pushErrorMessage(errorMessage, InvalidCompilerOption,
            "missing file name after -o option");
      outputBasenameStr = flagsVectorStr[i + 1];
    }
    onnxMlirCompile.appendStr(flagsVectorStr[i]);
  }
  // Push also the input file name.
  onnxMlirCompile.appendStr(inputFilenameStr);
  // Run command.
  int rc = onnxMlirCompile.exec();
  if (rc == CompilerSuccess && outputFilename) {
    // Determine output file name value and copy into outputFilename.
    if (outputBasenameStr.empty())
      outputBasenameStr =
          inputFilenameStr.substr(0, inputFilenameStr.find_last_of("."));
    std::string name = getTargetFilename(outputBasenameStr, emissionTarget);
    *outputFilename = strdup(name.c_str());
  }
  return rc != 0 ? CompilerFailureInLLVMOpt : CompilerSuccess;
}

#if 0 // hi alex
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
#endif

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
} // namespace onnx_mlir
