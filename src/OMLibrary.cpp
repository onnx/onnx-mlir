/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/OMLibrary.hpp"
#include "src/MainUtils.hpp"

extern "C" {
namespace onnx_mlir {
ONNX_MLIR_EXPORT int CompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mtriple) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;

  if (mtriple) {
    std::string mtripleString(mtriple);
    setTargetTriple(mtripleString);
  }

  registerDialects(context);
  processInputFile(std::string(inputFilename), context, module);
  return compileModule(module, context, outputBaseName, emissionTarget);
}

ONNX_MLIR_EXPORT int CompileFromArray(const void *inputBuffer, int bufferSize,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mtriple) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;

  if (mtriple) {
    std::string mtripleString(mtriple);
    setTargetTriple(mtripleString);
  }

  registerDialects(context);
  processInputArray(inputBuffer, bufferSize, context, module);
  return compileModule(module, context, outputBaseName, emissionTarget);
}

} // namespace onnx_mlir
}
