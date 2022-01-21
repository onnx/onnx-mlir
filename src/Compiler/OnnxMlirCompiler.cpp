/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnnxMlirCompiler.h"
#include "CompilerUtils.hpp"

extern "C" {
namespace onnx_mlir {
ONNX_MLIR_EXPORT int omCompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mcpu, const char *march, const char *mtriple,
    const OptLevel *optLevel, const char **errorMessage) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;

  setCompileContext(context, mcpu, march, mtriple, optLevel);
  std::string error_message;
  processInputFile(std::string(inputFilename), context, module, &error_message);
  if (errorMessage != NULL) {
    *errorMessage = error_message.c_str();
    return 1;
  }
  return compileModule(module, context, outputBaseName, emissionTarget);
}

ONNX_MLIR_EXPORT int omCompileFromArray(const void *inputBuffer, int bufferSize,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mcpu, const char *march, const char *mtriple,
    const OptLevel *optLevel) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;

  setCompileContext(context, mcpu, march, mtriple, optLevel);
  processInputArray(inputBuffer, bufferSize, context, module);
  return compileModule(module, context, outputBaseName, emissionTarget);
}

} // namespace onnx_mlir
}
