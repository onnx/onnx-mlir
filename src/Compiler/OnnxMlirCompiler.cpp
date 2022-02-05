/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnnxMlirCompiler.h"
#include "CompilerUtils.hpp"

extern "C" {
namespace onnx_mlir {

ONNX_MLIR_EXPORT OMCompilerOptions *omCreateCompilerOptions() {
  return new OMCompilerOptions;
}

ONNX_MLIR_EXPORT void omDestroyCompilerOptions(OMCompilerOptions *options) {
  if (options)
    delete options;
}

ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromEnv(
    OMCompilerOptions *options) {
  if (!options)
    return 1;
  return options->setFromEnv();
}

ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromArgs(
    OMCompilerOptions *options, int64_t argc, char *argv[]) {
  if (!options)
    return 1;
  return options->setFromArgs(argc, argv);
}

ONNX_MLIR_EXPORT int64_t omSetCompilerOptions(
    OMCompilerOptions *options, const OptionKind kind, const char *val) {
  if (!options)
    return 1;
  return options->set(kind, val)
}

ONNX_MLIR_EXPORT int64_t omGetUnusedCompilerOptionsArgs(
    OMCompilerOptions *options, int64_t *argc, char ***argv)
    {
  if (!options)
    return 1;
  return options->getUnusedArgs(*argc, argv);
}


ONNX_MLIR_EXPORT int omCompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const OMCompilerOptions *options, const char **errorMessage) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;

  if (options) options->setCompileContext(context);
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
    const OMCompilerOptions *options) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;

  if (options) options->setCompileContext(context);
  processInputArray(inputBuffer, bufferSize, context, module);
  return compileModule(module, context, outputBaseName, emissionTarget);
}

} // namespace onnx_mlir
}
