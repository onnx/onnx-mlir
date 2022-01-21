/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnnxMlirCompiler.h"
#include "CompilerUtils.hpp"

void setCompileContext(mlir::MLIRContext &context, const char *mcpu,
    const char *march, const char *mtriple) {
  if (mcpu)
    setTargetCPU(std::string(mcpu));
  if (march)
    setTargetArch(std::string(march));
  if (mtriple)
    setTargetTriple(std::string(mtriple));

  registerDialects(context);
}

extern "C" {
namespace onnx_mlir {
ONNX_MLIR_EXPORT int omCompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mcpu, const char *march, const char *mtriple,
    const OptLevel *optLevel, const char **errorMessage) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;

  setCompileContext(context, mcpu, march, mtriple);
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

  setCompileContext(context, mcpu, march, mtriple);
  processInputArray(inputBuffer, bufferSize, context, module);
  return compileModule(module, context, outputBaseName, emissionTarget);
}

} // namespace onnx_mlir
}
