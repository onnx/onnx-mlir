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

// Include pthreads (need special treatment on z/OS).
#ifdef __MVS__
#define _OPEN_THREADS
#endif
#include <pthread.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "src/Accelerators/NNPA/Runtime/zDNNExtension/zDNNExtension.h"
#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mmStruct_t {
  const zdnn_ztensor *A;
  const zdnn_ztensor *B;
  const zdnn_ztensor *C;
  int opType;
  zdnn_ztensor *O;
} mmStruct_t;

void *call_zdnn_matmul_op(void *args) {
  struct mmStruct_t *p = (struct mmStruct_t *)args;
  zdnn_status status =
      zdnn_matmul_op(p->A, p->B, p->C, (zdnn_matmul_ops)p->opType, p->O);
  return (void *)(__intptr_t)status;
}

void *call_zdnn_matmul_bcast_op(void *args) {
  struct mmStruct_t *p = (struct mmStruct_t *)args;
  zdnn_status status = zdnn_matmul_bcast_op(
      p->A, p->B, p->C, (zdnn_matmul_bcast_ops)p->opType, p->O);
  return (void *)(__intptr_t)status;
}

zdnn_status zdnn_matmul_op_common(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output, bool isBcast) {
  zdnn_status status;
  bool isDebug = ZTensorSplitDebugFromEnv();
  const zdnn_tensor_desc *descA = inputA->transformed_desc;
  const zdnn_tensor_desc *descB = inputB->transformed_desc;
  const zdnn_tensor_desc *descC = inputC->transformed_desc;
  const zdnn_tensor_desc *descO = output->transformed_desc;
  if (isDebug) {
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

  uint32_t dimSize = descA->dim2;
  uint32_t chunkSize = ZTensorSplitSizeFromEnv();
  uint32_t chunkSizeInStick = CEIL(chunkSize, AIU_STICKS_PER_PAGE);
  uint32_t numOfChunks = CEIL(dimSize, chunkSize);
  if (isDebug)
    printf("The number of chunks: %d\n", numOfChunks);

  // Dim is small or ztensor split is disabled.
  if (numOfChunks == 1 || !ZTensorSplitEnabledFromEnv()) {
    if (isDebug)
      printf("Not split zTensor ...\n");
    mmStruct_t args = {
        .A = inputA, .B = inputB, .C = inputC, .opType = opType, .O = output};
    zdnn_status status;
    if (isBcast)
      status =
          (zdnn_status)(__intptr_t)call_zdnn_matmul_bcast_op((void *)&args);
    else
      status = (zdnn_status)(__intptr_t)call_zdnn_matmul_op((void *)&args);
    return status;
  }

  if (isDebug)
    printf("Split zTensor A ...\n");

  // Split input A and do matmul.
  double splitTime = 0.;
  double mmTime = 0.;
  double mergeTime = 0.;
  clock_t start_time, end_time;
  for (uint32_t i = 0; i < numOfChunks; ++i) {
    uint32_t offset = i * chunkSizeInStick;
    // Adjust chunkSize for the last chunk.
    if (i == numOfChunks - 1)
      chunkSize = dimSize - i * chunkSize;

    if (isDebug)
      printf("Processing chunk %d of size %d \n", i, chunkSize);

    // Prepare input and output chunks.
    if (isDebug)
      start_time = clock();
    zdnn_ztensor *zaTensor = malloc(sizeof(struct zdnn_ztensor));
    zdnn_ztensor *zoTensor = malloc(sizeof(struct zdnn_ztensor));
    allocZTensorInDim2(inputA, chunkSize, zaTensor);
    allocZTensorInDim2(output, chunkSize, zoTensor);
    copyZTensorInDim2(zaTensor, inputA, offset, /*fromChunk=*/false);
    if (isDebug) {
      end_time = clock();
      splitTime +=
          ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
    }

    if (isDebug) {
      zTensorShape shape;
      getZTensorShape(zaTensor, &shape);
      printf("zTensor input chunk %d: [%d, %d, %d, %d, %d, %d]\n", i,
          shape.dim6, shape.dim5, shape.dim4, shape.dim3, shape.dim2,
          shape.dim1);
      getZTensorShape(zoTensor, &shape);
      printf("zTensor output chunk %d: [%d, %d, %d, %d, %d, %d]\n", i,
          shape.dim6, shape.dim5, shape.dim4, shape.dim3, shape.dim2,
          shape.dim1);
    }

    // Call zdnn_matmul_op on the chunk.
    if (isDebug)
      start_time = clock();
    mmStruct_t args = {.A = zaTensor,
        .B = inputB,
        .C = inputC,
        .opType = opType,
        .O = zoTensor};
    if (isBcast)
      status =
          (zdnn_status)(__intptr_t)call_zdnn_matmul_bcast_op((void *)&args);
    else
      status = (zdnn_status)(__intptr_t)call_zdnn_matmul_op((void *)&args);
    assert(status == ZDNN_OK);
    if (isDebug) {
      end_time = clock();
      mmTime += ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
    }

    if (isDebug)
      start_time = clock();
    copyZTensorInDim2(output, zoTensor, offset, /*fromChunk=*/true);
    if (isDebug) {
      end_time = clock();
      mergeTime +=
          ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
    }

    // Free the chunks.
    status = freeZTensorChunk(zaTensor);
    assert(status == ZDNN_OK);
    status = freeZTensorChunk(zoTensor);
    assert(status == ZDNN_OK);
  }

  if (isDebug)
    printf("split: %f, mm: %f, merge: %f (milliseconds)\n", splitTime, mmTime,
        mergeTime);

  // struct mmStruct_t args = {
  //     .A = inputA, .B = inputB, .C = inputC, .opType = opType, .O = output};
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

zdnn_status zdnn_matmul_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output) {
  return zdnn_matmul_op_common(
      inputA, inputB, inputC, opType, output, /*isBcast=*/false);
}

zdnn_status zdnn_matmul_bcast_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output) {
  return zdnn_matmul_op_common(
      inputA, inputB, inputC, opType, output, /*isBcast=*/true);
}

#ifdef __cplusplus
}
#endif
