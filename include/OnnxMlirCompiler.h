/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- OnnxMlirCompiler.h - ONNX-MLIR Compiler API Declarations -----===//
//
// This file contains declaration of onnx-mlir compiler functionality
// exported from the OnnxMlirCompiler library
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNXMLIRCOMPILER_H
#define ONNX_MLIR_ONNXMLIRCOMPILER_H

#include <onnx-mlir/Compiler/OMCompilerTypes.h>

#ifdef ONNX_MLIR_BUILT_AS_STATIC
#define ONNX_MLIR_EXPORT
#define ONNX_MLIR_NO_EXPORT
#else
#ifdef _MSC_VER
#ifdef OnnxMlirCompiler_EXPORTS
/* We are building this library */
#define ONNX_MLIR_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define ONNX_MLIR_EXPORT __declspec(dllimport)
#endif
#else
#define ONNX_MLIR_EXPORT __attribute__((__visibility__("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

namespace onnx_mlir {

/*!
 *  Compile an onnx model from a file containing MLIR or ONNX protobuf.
 *  @param inputFilename File name pointing onnx model protobuf or MLIR.
 *  @param outputBaseName File name without extension to write output.
 *  @param emissionTarget Target format to compile to.
 *  @param optionKey List of keys specified for the compiler options.
 *  @param optionVal List of string values for corresponding keys.
 *  @param optionNum Number of keys and strings.
 *  @param errorMessage Error message.
 *  @return 0 on success or non-zero error code on failure.
 */
ONNX_MLIR_EXPORT int omCompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const onnx_mlir::OptionKind *optionKey, const char **optionVal,
    const int optionNum, const char **errorMessage);

/*!
 *  Compile an onnx model from an ONNX protobuf array.
 *  @param inputBuffer ONNX protobuf array.
 *  @param bufferSize Size of ONNX protobuf array.
 *  @param outputBaseName File name without extension to write output.
 *  @param emissionTarget Target format to compile to.
 *  @param optionKey List of keys specified for the compiler options.
 *  @param optionVal List of string values for corresponding keys.
 *  @param optionNum Number of keys and strings.
 *  @return 0 on success or non-zero error code on failure
 */
ONNX_MLIR_EXPORT int omCompileFromArray(const void *inputBuffer, int bufferSize,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const onnx_mlir::OptionKind *optionKey, const char **optionVal,
    const int optionNum
);

} // namespace onnx_mlir

#ifdef __cplusplus
}
#endif

#endif
