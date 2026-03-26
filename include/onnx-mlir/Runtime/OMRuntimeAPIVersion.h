/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ OMRuntimeAPIVersion.h ---------------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// Runtime API version information.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_RUNTIME_OMRUNTIMEAPIVERSION_H
#define ONNX_MLIR_RUNTIME_OMRUNTIMEAPIVERSION_H

#ifdef __cplusplus
extern "C" {
#endif

#define ONNX_MLIR_RUNTIME_API_VERSION_MAJOR 1
#define ONNX_MLIR_RUNTIME_API_VERSION_MINOR 0
#define ONNX_MLIR_RUNTIME_API_VERSION_PATCH 0

#define ONNX_MLIR_RUNTIME_API_VERSION \
  ((ONNX_MLIR_RUNTIME_API_VERSION_MAJOR << 16) | \
   (ONNX_MLIR_RUNTIME_API_VERSION_MINOR << 8) | \
   ONNX_MLIR_RUNTIME_API_VERSION_PATCH)

const char *omGetRuntimeAPIVersion(void);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_RUNTIME_OMRUNTIMEAPIVERSION_H