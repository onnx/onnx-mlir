/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- MatMul.c ----------------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// A wrapper of zdnn_matmul_op for ztensor partition and parallelism.
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

#include "zDNNExtension.h"
#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline zdnn_status call_zdnn_matmul_op(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output, bool isBcast) {
  if (isBcast)
    return zdnn_matmul_bcast_op(
        inputA, inputB, inputC, (zdnn_matmul_bcast_ops)opType, output);
  return zdnn_matmul_op(
      inputA, inputB, inputC, (zdnn_matmul_ops)opType, output);
}

static zdnn_status zdnn_matmul_op_common(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output, bool isBcast) {
  double totalTime = 0.;
  clock_t start_time = 0, end_time = 0;

  if (OMZTensorSplitDebug)
    start_time = clock();

  // Verify that e4, e3 do not exceed the maximum dimension size. Thus, we
  // will split e2 and e1 safely.
  OrigShape origShapeOfA, origShapeOfB, origShapeOfC;
  getOrigShape(inputA, &origShapeOfA);
  getOrigShape(inputB, &origShapeOfB);
  getOrigShape(inputC, &origShapeOfC);
  if (OMZTensorSplitDebug) {
    printf("[MatMul] A:  e4 = %d, e3 = %d, e2 = %d, e1 = %d.\n",
        origShapeOfA.e4, origShapeOfA.e3, origShapeOfA.e2, origShapeOfA.e1);
    printf("[MatMul] B:  e4 = %d, e3 = %d, e2 = %d, e1 = %d.\n",
        origShapeOfA.e4, origShapeOfB.e3, origShapeOfB.e2, origShapeOfB.e1);
    printf("[MatMul] C:  e4 = %d, e3 = %d, e2 = %d, e1 = %d.\n",
        origShapeOfA.e4, origShapeOfC.e3, origShapeOfC.e2, origShapeOfC.e1);
  }
  uint32_t maxDimSize = zdnn_get_nnpa_max_dim_idx_size();
  if ((origShapeOfA.e4 > maxDimSize) || (origShapeOfA.e3 > maxDimSize)) {
    printf("[MatMul] The 1st tensor dimension exceeds maximum dimension index "
           "size (MDIS) of %d: e4 = %d, e3 = %d.\n",
        maxDimSize, origShapeOfA.e4, origShapeOfA.e3);
    return ZDNN_EXCEEDS_MDIS;
  }
  if ((origShapeOfB.e4 > maxDimSize) || (origShapeOfB.e3 > maxDimSize)) {
    printf("[MatMul] The 2nd tensor dimension exceeds maximum dimension index "
           "size (MDIS) of %d: e4 = %d, e3 = %d.\n",
        maxDimSize, origShapeOfB.e4, origShapeOfB.e3);
    return ZDNN_EXCEEDS_MDIS;
  }

  // For a MatMul of A(M,N)*B(N,P)+C(P),
  // We split M that is e2 in (e4, e3, e2, e1), and P that is e1.
  SplitInfo splitInfoA = {
      .origZTensor = inputA, .axis = 2, .chunkSize = OMZTensorSplitSize};
  SplitInfo splitInfoB = {
      .origZTensor = inputB, .axis = 3, .chunkSize = OMZTensorSplitSize};
  SplitInfo splitInfoC = {
      .origZTensor = inputC, .axis = 3, .chunkSize = OMZTensorSplitSize};
  SplitInfo splitInfoY = {
      .origZTensor = output, .axis = 2, .chunkSize = OMZTensorSplitSize};

  initSplitInfo(&splitInfoA);
  initSplitInfo(&splitInfoB);
  initSplitInfo(&splitInfoC);
  initSplitInfo(&splitInfoY);

  // Split input A into chunks.
  if (OMZTensorSplitDebug)
    printf("[MatMul] Split the 1st ztensor (A) along e2 into %d chunks of %d "
           "elements. ReuseZTensor: %d, ReuseBuffer: %d \n",
        splitInfoA.numOfChunks, splitInfoA.chunkSize,
        splitInfoA.reuseOrigZTensor, splitInfoA.reuseOrigBuffer);
  splitZTensor(&splitInfoA, /*copyData=*/true);
  splitZTensor(&splitInfoY, /*copyData=*/false);
  // Split input B and C into chunks.
  if (OMZTensorSplitDebug) {
    printf("[MatMul] Split the 2nd ztensor (B) along e1 into %d chunks of %d "
           "elements. ReuseZTensor: %d, ReuseBuffer: %d \n",
        splitInfoB.numOfChunks, splitInfoB.chunkSize,
        splitInfoB.reuseOrigZTensor, splitInfoB.reuseOrigBuffer);
    printf("[MatMul] Split the 3rd ztensor (C) along e1 into %d chunks of %d "
           "elements. ReuseZTensor: %d, ReuseBuffer: %d \n",
        splitInfoC.numOfChunks, splitInfoC.chunkSize,
        splitInfoC.reuseOrigZTensor, splitInfoC.reuseOrigBuffer);
  }
  splitZTensor(&splitInfoB, /*copyData=*/true);
  splitZTensor(&splitInfoC, /*copyData=*/true);

  // Call zdnn_matmul_op on each chunk.
  // Iterate over the chunks along the first dim of A.
  for (uint32_t i = 0; i < splitInfoA.numOfChunks; ++i) {
    zdnn_ztensor *zaTensor = (splitInfoA.chunks + i)->ztensor;
    zdnn_ztensor *zyTensor = (splitInfoY.chunks + i)->ztensor;

    SplitInfo splitInfoYB = {
        .origZTensor = zyTensor, .axis = 3, .chunkSize = OMZTensorSplitSize};
    initSplitInfo(&splitInfoYB);
    splitZTensor(&splitInfoYB, /*copyData=*/false);
    // Iterate over the chunks along the second dim of B.
    for (uint32_t j = 0; j < splitInfoB.numOfChunks; ++j) {
      zdnn_ztensor *zbTensor = (splitInfoB.chunks + j)->ztensor;
      zdnn_ztensor *zcTensor = (splitInfoC.chunks + j)->ztensor;
      zdnn_ztensor *zybTensor = (splitInfoYB.chunks + j)->ztensor;
      zdnn_status status = call_zdnn_matmul_op(
          zaTensor, zbTensor, zcTensor, opType, zybTensor, isBcast);
      assert(status == ZDNN_OK);
    }
    mergeZTensors(&splitInfoYB);
    freeSplitInfoBuffer(&splitInfoYB);
  }

  // Merging the chunks into the output.
  mergeZTensors(&splitInfoY);
  freeSplitInfoBuffer(&splitInfoA);
  freeSplitInfoBuffer(&splitInfoB);
  freeSplitInfoBuffer(&splitInfoC);
  freeSplitInfoBuffer(&splitInfoY);

  if (OMZTensorSplitDebug) {
    end_time = clock();
    totalTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
    printf("[MatMul] total time, %f (milliseconds)\n", totalTime);
  }

  return ZDNN_OK;
}

// -----------------------------------------------------------------------------
// Extension Functions
// Same name as zdnn functions but with the `_ext` postfix.
// -----------------------------------------------------------------------------

zdnn_status zdnn_matmul_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output) {
  return zdnn_matmul_op_common(
      inputA, inputB, inputC, opType, output, /*isBcast=*/false);
}

zdnn_status zdnn_matmul_bcast_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output) {
  zdnn_status status = zdnn_matmul_op_common(
      inputA, inputB, inputC, opType, output, /*isBcast=*/true);
  // Compiler does not check the return result at this moment. Thus, check it
  // here.
  assert(status == ZDNN_OK && "Failed to execute MatMul on NNPA");
  return status;
}

#ifdef __cplusplus
}
#endif
