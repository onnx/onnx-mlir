/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMNNPA.c ---------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Onnx MLIR NNPA Accelerator Runtime
//
//===----------------------------------------------------------------------===//

#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

// Required name: InitAccelX where X=NNPA here.
void InitAccelNNPA() { zdnn_init(); }

// Required name: ShutdownAccelX where X=NNPA here.
void ShutdownAccelNNPA() {}

#ifdef __cplusplus
}
#endif
