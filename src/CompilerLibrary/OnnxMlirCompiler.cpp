/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnnxMlirCompiler.h"
#include "src/MainUtils.hpp"

extern "C" {
namespace onnx_mlir {
ONNX_MLIR_EXPORT int CompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mcpu, const char *mtriple) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;
  std::string mcpuString;
  std::string mtripleString;

  if (mcpu) {
    mcpuString = std::string(mcpu);
    setTargetCPU(mcpuString);
  }

  if (mtriple) {
    mtripleString = std::string(mtriple);
    setTargetTriple(mtripleString);
  }

  registerDialects(context);
  processInputFile(std::string(inputFilename), context, module);
  return compileModule(module, context, outputBaseName, emissionTarget);
}

ONNX_MLIR_EXPORT int CompileFromArray(const void *inputBuffer, int bufferSize,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mcpu, const char *mtriple) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;

  std::string mcpuString;
  std::string mtripleString;

  if (mcpu != nullptr) {
    mcpuString = std::string(mcpu);
    setTargetCPU(mcpuString);
  }

  if (mtriple != nullptr) {
    mtripleString = std::string(mtriple);
    setTargetTriple(mtripleString);
  }

  registerDialects(context);
  processInputArray(inputBuffer, bufferSize, context, module);
  return compileModule(module, context, outputBaseName, emissionTarget);
}

} // namespace onnx_mlir
}
