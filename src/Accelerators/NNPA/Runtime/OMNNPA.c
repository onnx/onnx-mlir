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

void InitAccelNNPA() {
  zdnn_init();
}

void ShutdownAccelNNPA() {}

#ifdef __cplusplus
}
#endif
