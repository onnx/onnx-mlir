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
 *  Compile an onnx model from a file containing MLIR or ONNX protobuf
 *  @param inputFilename File name pointing onnx model protobuf or MLIR
 *  @param outputBaseName File name without extension to write output
 *  @param emissionTarget Target format to compile to
 *  @param mcpu Optional target CPU
 *  @param mtriple Optional target architecture
 *  @return 0 on success or non-zero error code on failure
 */
ONNX_MLIR_EXPORT int omCompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mcpu, const char *mtriple, const char **errorMessage);

/*!
 *  Compile an onnx model from an ONNX protobuf array
 *  @param inputBuffer ONNX protobuf array
 *  @param bufferSize Size of ONNX protobuf array
 *  @param outputBaseName File name without extension to write output
 *  @param emissionTarget Target format to compile to
 *  @param mcpu Optional target CPU
 *  @param mtriple Optional compile target triple
 *  @return 0 on success or non-zero error code on failure
 */
ONNX_MLIR_EXPORT int omCompileFromArray(const void *inputBuffer, int bufferSize,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mcpu, const char *mtriple);

} // namespace onnx_mlir

#ifdef __cplusplus
}
#endif

#endif
