/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMMatMulAsyncNNPA.h
//------------------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// Onnx MatMul op on NNPA Accelerator Runtime
//
//===----------------------------------------------------------------------===//

// Include pthreads (need special treatment on Zos).
#ifdef __MVS__
#define _OPEN_THREADS
#endif
#include <pthread.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "onnx-mlir/Runtime/OMTensor.h"
#include "onnx-mlir/Runtime/OnnxDataType.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMTHREADHANDER_SIZE 64
typedef struct _OMThreadHandler {
  pthread_t threadID;
  struct _threadArgs {
    char _researved[OMTHREADHANDER_SIZE];
  } threadArgs;
} OMThreadHandler;

//
// Calculate matrix multiplication asynchronously: Y = A * B
// omTensorAsyncWait need to be called before asscsing the results.
//
void omTensorMatMulAsync(
    OMTensor *Y, OMTensor *threadTensor, OMTensor *A, OMTensor *B, OMTensor *C);

//
// Wait completion of the corresponding omTensorMatMulAsync call.
// It need to be called before asscsing the results.
//
void omTensorAsyncWait(OMTensor *threadTensor);

#ifdef __cplusplus
}
#endif
