/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- MatMul.c ----------------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// A wrapper of zdnn_matmul_op supports very large input tensors.
//
//===----------------------------------------------------------------------===//

// Include pthreads (need special treatment on Zos).
#ifdef __MVS__
#define _OPEN_THREADS
#endif
#include <pthread.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "src/Accelerators/NNPA/Runtime/zDNNExtension/zDNNExtension.h"
#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG 1
#define USE_PTHREAD 1

typedef struct mmStruct {
  const zdnn_ztensor *A;
  const zdnn_ztensor *B;
  const zdnn_ztensor *C;
  zdnn_matmul_ops opType;
  zdnn_ztensor *O;
} mmStruct;

void *call_zdnn_matmul_op(void *args) {
  struct mmStruct *p = (struct mmStruct *)args;
  zdnn_status status = zdnn_matmul_op(p->A, p->B, p->C, p->opType, p->O);
  return (void *)(__intptr_t)status;
}

zdnn_status zdnn_matmul_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC,
    zdnn_matmul_ops op_type, zdnn_ztensor *output) {
  zdnn_status status;
  // inputA->transformed_desc->dim2 = 8;
  // output->transformed_desc->dim2 = 8;
  const zdnn_tensor_desc *descA = inputA->transformed_desc;
  const zdnn_tensor_desc *descB = inputB->transformed_desc;
  const zdnn_tensor_desc *descC = inputC->transformed_desc;
  const zdnn_tensor_desc *descO = output->transformed_desc;
  if (DEBUG) {
    printf("I am in zdnn_matmul_op_ext\n");
    printf("A: [%d, %d, %d, %d], %d, ", descA->dim4, descA->dim3, descA->dim2,
        descA->dim1, descA->layout);
    descB = inputB->transformed_desc;
    printf("B: [%d, %d, %d, %d], %d, ", descB->dim4, descB->dim3, descB->dim2,
        descB->dim1, descB->layout);
    descC = inputC->transformed_desc;
    printf("C: [%d, %d, %d, %d], %d.", descC->dim4, descC->dim3, descC->dim2,
        descC->dim1, descC->layout);
    descO = output->transformed_desc;
    printf("Output: [%d, %d, %d, %d], %d\n", descO->dim4, descO->dim3,
        descO->dim2, descO->dim1, descO->layout);
  }

  // Split along dim2 of A.
  zTensorShape zaShape;
  getZTensorShape(inputA, &zaShape);

  uint32_t numOfChunks = CEIL(descA->dim2, CHUNK_SIZE);

  // Small tensor, there is no partition.
  if (numOfChunks == 1) {
    printf("No partition\n");
    struct mmStruct args = {
        .A = inputA, .B = inputB, .C = inputC, .opType = op_type, .O = output};
    zdnn_status status =
        (zdnn_status)(__intptr_t)call_zdnn_matmul_op((void *)&args);
    return status;
  }

  // Split input A and do matmul.
  struct zdnn_ztensor *zoTensors =
      malloc(numOfChunks * sizeof(struct zdnn_ztensor));
  for (uint32_t i = 0; i < numOfChunks; ++i) {
    printf("chunk %d\n", i);
    bool isLast = (i == numOfChunks - 1);

    // Prepare a chunk of input A.
    zdnn_ztensor *zaTensor = malloc(sizeof(struct zdnn_ztensor));
    createZTensorInDim2(inputA, i, isLast, zaTensor);

    // Prepare a chunk of output 0.
    zdnn_ztensor *zoTensor = zoTensors + i;
    createZTensorInDim2(output, i, isLast, zoTensor);

    // Call zdnn_matmul_op on the chunk.
    struct mmStruct args = {.A = zaTensor,
        .B = inputB,
        .C = inputC,
        .opType = op_type,
        .O = zoTensor};
    status = (zdnn_status)(__intptr_t)call_zdnn_matmul_op((void *)&args);

    // just for test.
    // todo: remove this.
    zTensorShape zShape;
    getZTensorShape(zaTensor, &zShape);

    // Free the chunk of A.
    freeZTensorChunk(zaTensor, i != 0);
  }

  // Free the output chunks.
  for (uint32_t i = 0; i < numOfChunks; ++i) {
    zdnn_ztensor *t = zoTensors + i;
    freeZTensorChunk(t, i != 0);
  }
  free(zoTensors);

  // struct mmStruct args = {
  //     .A = inputA, .B = inputB, .C = inputC, .opType = op_type, .O = output};
  // if (USE_PTHREAD) {
  //   printf("Using pthread\n");
  //   pthread_t thread_id;
  //   pthread_create(&thread_id, NULL, call_zdnn_matmul_op, (void *)&args);
  //   pthread_join(thread_id, (void *)(__intptr_t)&status);
  // } else {
  //   printf("No pthread\n");
  //   status = (zdnn_status)(__intptr_t)call_zdnn_matmul_op((void *)&args);
  // }

  assert(status == ZDNN_OK && "Failed to call zdnn_matmul_op");
  return status;
}

#ifdef __cplusplus
}
#endif
