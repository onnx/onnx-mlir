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
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "llvm/Support/FileSystem.h"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

// Derive the name; base name is either given by a "-o" option, or is taken as
// the model name. The extention depends on the target; e.g. -EmitLib will
// generate a .so, other targets may generate a .mlir.
static std::string deriveOutputFileName(
    std::vector<std::string> &flagVect, std::string inputFilename) {
  // Get output file name.
  std::string outputBasename;
  int num = flagVect.size();
  for (int i = 0; i < num; ++i) {
    if (flagVect[i].find("-o=", 0, 3) == 0) {
      if (flagVect[i].length() > 3) {
        outputBasename = flagVect[i].substr(3);
        break;
      } else
        llvm::errs() << "Parsing `-o=` option, expected a name. Use default.\n";
    } else if (flagVect[i].find("-o") == 0) {
      if (i < num - 1) {
        outputBasename = flagVect[i + 1];
        break;
      } else
        llvm::errs() << "Parsing `-o` option, expected a name. Use default.\n";
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

static std::vector<std::string> parseFlags(const char *flags) {
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
  return flagVect;
}

extern "C" {

ONNX_MLIR_EXPORT int64_t omCompileFromFile(const char *inputFilename,
    const char *flags, char **outputFilename, char **errorMessage) {
  // Ensure known values in filename and error message if provided.
  if (outputFilename)
    *outputFilename = nullptr;
  if (errorMessage)
    *errorMessage = nullptr;

  // Process the flags, saving each space-separated text in a separate
  // entry in the string vector flagVect.
  std::vector<std::string> flagVect = parseFlags(flags);

  // Use 'onnx-mlir' command to compile the model.
  std::string onnxMlirPath;
  const auto &envDir = getEnvVar("ONNX_MLIR_BIN_PATH");
  if (envDir && llvm::sys::fs::exists(envDir.value()))
    onnxMlirPath = envDir.value() + "/onnx-mlir";
  else
    onnxMlirPath = getToolPath("onnx-mlir");
  Command onnxMlirCompile(onnxMlirPath);
  // Add flags and input flag.
  onnxMlirCompile.appendList(flagVect);
  std::string inputFilenameStr(inputFilename);
  onnxMlirCompile.appendStr(inputFilenameStr);
  // Run command.
  int rc = onnxMlirCompile.exec();
  if (rc != CompilerSuccess) {
    // Failure to compile.
    if (errorMessage) {
      std::string errorStr =
          "Compiler failed with error code " + std::to_string(rc);
      *errorMessage = strdup(errorStr.c_str());
    }
    return CompilerFailureInLLVMOpt;
  }
  // Success.
  if (outputFilename) {
    std::string name = deriveOutputFileName(flagVect, inputFilenameStr);
    *outputFilename = strdup(name.c_str());
  }
  return CompilerSuccess;
}

ONNX_MLIR_EXPORT int64_t omCompileFromArray(const void *inputBuffer,
    int64_t bufferSize, const char *outputBaseName,
    EmissionTargetType emissionTarget, char **outputFilename,
    char **errorMessage) {
  // Ensure known values in filename and error message if provided.
  if (outputFilename)
    *outputFilename = nullptr;
  if (errorMessage)
    *errorMessage = nullptr;

  OwningOpRef<ModuleOp> module;
  MLIRContext context;
  loadDialects(context);

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

ONNX_MLIR_EXPORT char *omCompileOutputFileName(
    const char *inputFilename, const char *flags) {
  std::vector<std::string> flagVect = parseFlags(flags);
  std::string inputFilenameStr(inputFilename);
  std::string name = deriveOutputFileName(flagVect, inputFilenameStr);
  return strdup(name.c_str());
}

ONNX_MLIR_EXPORT char *omCompileModelTag(const char *flags) {
  std::string modelTag = "";
  std::vector<std::string> flagVect = parseFlags(flags);
  for (int i = 0; i < (int)flagVect.size(); ++i) {
    if (flagVect[i].find("--tag=") == 0) {
      modelTag = flagVect[i].substr(6);
      break;
    }
    if (flagVect[i].find("-tag=") == 0) {
      modelTag = flagVect[i].substr(5);
      break;
    }
  }
  return strdup(modelTag.c_str());
}

} // extern C
} // namespace onnx_mlir
