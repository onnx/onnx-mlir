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

namespace onnx_mlir {

static std::string deriveOutputFileName(
    std::vector<std::string> &flagVect, std::string inputFilename) {
  // Get output file name.
  std::string outputBasename;
  int num = flagVect.size();
  for (int i = 0; i < num - 1;
       ++i) { // Skip last as need 2 consecutive entries.
    if (flagVect[i].find("-o") == 0) {
      outputBasename = flagVect[i + 1];
      break;
    }
  }
  // If no output file name, derive it from input file name
  if (outputBasename.empty())
    outputBasename = inputFilename.substr(0, inputFilename.find_last_of("."));
  // Get Emit target (approximate, enough to get output name). There are many
  // more Emit target than in the base definition because Accelerators may
  // have their own. That is why the Emit target has to be part of the flags
  // and cannot be a direct enum, as there is none that encompass all the
  // possible options.
  EmissionTargetType emissionTarget = EmissionTargetType::EmitLib;
  for (int i = 0; i < num; ++i) {
    if (flagVect[i].find("-Emit") == 0 || flagVect[i].find("--Emit") == 0) {
      if (flagVect[i].find("Lib") <= 6)
        emissionTarget = EmissionTargetType::EmitLib;
      else if (flagVect[i].find("JNI") <= 6)
        emissionTarget = EmissionTargetType::EmitJNI;
      else if (flagVect[i].find("Obj") <= 6)
        emissionTarget = EmissionTargetType::EmitObj;
      else // There are many other targets, all of the MLIR type.
        emissionTarget = EmissionTargetType::EmitMLIR;
      break;
    }
  }
  // Derive output file name from base and emission target.
  return getTargetFilename(outputBasename, emissionTarget);
}

extern "C" {

ONNX_MLIR_EXPORT int64_t omCompileFromFile(const char *inputFilename,
    const char *flags, const char **outputFilename, const char **errorMessage) {
  // Process the flags, saving each space-separated text in a separate
  // entry in the string vector flagVect.
  std::vector<std::string> flagVect;
  const char *str = flags;
  do {
    // Get rid of leading spaces.
    while (*str && std::isspace(*str))
      ++str;
    // Save current location and advance while useful chars.
    const char *begin = str;
    while (*str && !std::isspace(*str))
      ++str;
    // If not empty, copy new entry into flagVec.
    if (begin != str)
      flagVect.push_back(std::string(begin, str));
  } while (*str);
  // Use 'onnx-mlir' command to compile the model.
  std::string onnxMlirPath;
  const auto &envDir = getEnvVar("ONNX_MLIR_BIN_PATH");
  if (envDir && llvm::sys::fs::exists(envDir.value()))
    onnxMlirPath = envDir.value() + "/onnx-mlir";
  else
    onnxMlirPath = getToolPath("onnx-mlir", kOnnxmlirPath);
  Command onnxMlirCompile(onnxMlirPath);
  // Add flags and input flag.
  onnxMlirCompile.appendList(flagVect);
  std::string inputFilenameStr(inputFilename);
  onnxMlirCompile.appendStr(inputFilenameStr);
  // Run command.
  int rc = onnxMlirCompile.exec();
  if (rc == CompilerSuccess && outputFilename) {
    std::string name = deriveOutputFileName(flagVect, inputFilenameStr);
    *outputFilename = strdup(name.c_str());
  }
  return rc != 0 ? CompilerFailureInLLVMOpt : CompilerSuccess;
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

} // extern C
} // namespace onnx_mlir
