/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "OMExport.hpp"
#include "OMTypes.hpp"

extern "C" {
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
ONNX_MLIR_EXPORT int CompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mcpu, const char *mtriple);

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
ONNX_MLIR_EXPORT int CompileFromArray(const void *inputBuffer, int bufferSize,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char *mcpu, const char *mtriple);

} // namespace onnx_mlir
}
