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
#include <string>
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif // #ifdef __cplusplus

#ifdef ONNX_MLIR_BUILT_AS_STATIC
#define ONNX_MLIR_EXPORT
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
namespace onnx_mlir {
#endif

// Preserve this file (but not included in onnx-mlir/include) just in case we
// need to reuse the macro being defined above, might be useful in the future.

#ifdef __cplusplus
} // namespace onnx_mlir
} // extern C
#endif

#endif
