/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- OMAsyncMatMul.cpp - OMTensor C/C++ Implementation ----------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ neutral implementation of OMTensorList data
// structures and helper functions.
//
//===----------------------------------------------------------------------===//

#ifdef __MVS__
#define _OPEN_THREADS
#endif

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"
#include "onnx-mlir/Runtime/OnnxDataType.h"

#define MAX_THREAD_NUM 2

#ifdef __cplusplus
extern "C" {
#endif

struct run_matmul_cpu_args {
  void *dataA;
  void *dataB;
  void *dataY;
  int64_t dim_m;
  int64_t dim_n;
  int64_t dim_p;
};

void *run_matmul_cpu(void *_args) {
  struct run_matmul_cpu_args *args = (struct run_matmul_cpu_args *)_args;
  void *dataA = args->dataA;
  void *dataB = args->dataA;
  void *dataY = args->dataY;
  int64_t dim_m = args->dim_m;
  int64_t dim_n = args->dim_n;
  int64_t dim_p = args->dim_p;

  for (int64_t m = 0; m < dim_m; ++m) {
    for (int64_t p = 0; p < dim_p; ++p) {
      float *y = (float *)dataY + m * dim_p + p;
      *y = 0;
      for (int64_t n = 0; n < dim_n; n++) {
        float *tmpA = (float *)dataA + m * dim_n + n;
        float *tmpB = (float *)dataB + n * dim_p + p;
        *y += (*tmpA) * (*tmpB);
      }
    }
  }
}

void omTensorMatMulAsync(
    OMTensor *Y, OMTensor *token, OMTensor *A, OMTensor *B, OMTensor *C) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(A);
  assert(dataType == ONNX_TYPE_FLOAT &&
         "omTensorMatmul assumes ONNX_TYPE_FLOAT type");
  assert((omTensorGetRank(A) == 2) && (omTensorGetRank(A) == 2) &&
         (omTensorGetRank(A) == 2) && "omTensorMatmul assumes rank 2 tensors");
  const int64_t *shapeA = omTensorGetShape(A);
  const int64_t *shapeB = omTensorGetShape(B);
  const int64_t *shapeC = omTensorGetShape(C);
  const int64_t *shapeY = omTensorGetShape(Y);
  void *dataA = omTensorGetDataPtr(A);
  void *dataB = omTensorGetDataPtr(B);
  void *dataC = omTensorGetDataPtr(C);
  void *dataY = omTensorGetDataPtr(Y);
  assert(shapeA[1] == shapeB[0] && "omTensorMatmul: inconsistent input shapes");
  int64_t dim_m = shapeA[0];
  int64_t dim_n = shapeA[1];
  int64_t dim_p = shapeB[1];

  struct run_matmul_cpu_args args = {dataA, dataB, dataY, dim_m, dim_n, dim_p};

  // Compute Matmul A(mxn) * B(nxp).
  // run_matmul_cpu((void *) &args);
  void *tokenPtr = omTensorGetDataPtr(token);
  uint64_t *threadID = (uint64_t *)tokenPtr;
  pthread_create((pthread_t *)threadID, NULL, run_matmul_cpu, (void *)&args);
}

void omTensorAsyncWait(OMTensor *token) {
  void *tokenPtr = omTensorGetDataPtr(token);
  uint64_t *threadID = (uint64_t *)tokenPtr;
  pthread_join(*((pthread_t *)threadID), NULL);
}

#ifdef __cplusplus
}
#endif

#if 0
//
// Calculate matrix multiplication asynchronously: Y = A * B
// omTensorAsyncWait need to be called before asscsing the results.
//
void omTensorMatMulAsync(OMTensor *Y, uint64_t *threadID, OMTensor *A, OMTensor *B,
                         OMTensor *C) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(A);
  assert(dataType == ONNX_TYPE_FLOAT &&
         "omTensorMatmul assumes ONNX_TYPE_FLOAT type");
  assert((omTensorGetRank(A) == 2) && (omTensorGetRank(A) == 2) &&
         (omTensorGetRank(A) == 2) && "omTensorMatmul assumes rank 2 tensors");
  const int64_t *shapeA = omTensorGetShape(A);
  const int64_t *shapeB = omTensorGetShape(B);
  const int64_t *shapeC = omTensorGetShape(C);
  const int64_t *shapeY = omTensorGetShape(Y);
  void *dataA = omTensorGetDataPtr(A);
  void *dataB = omTensorGetDataPtr(B);
  void *dataC = omTensorGetDataPtr(C);
  void *dataY = omTensorGetDataPtr(Y);
  void *dataToken = omTensorGetDataPtr(Token);
  assert(shapeA[1] == shapeB[0] && "omTensorMatmul: inconsistent input shapes");
  int dim_m = shapeA[0];
  int dim_n = shapeA[1];
  int dim_p = shapeB[1];

  struct run_matmul_op_cpu_args args = {
      dataA, dataB, dataY, dim_m, dim_n, dim_p};

  uint64_t *threadID;
  pthread_create((pthread_t *) threadID, NULL, run_matmul_op_cpu,
        (void *) &args);

  *dataToken = *threadID;

}

void omTensorAsyncWait(OMTensor *threadID) {
  //  pthread_join((pthread_t *) threadID, NULL);
}

