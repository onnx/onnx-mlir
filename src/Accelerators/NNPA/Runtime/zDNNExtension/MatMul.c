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
  clock_t start_time, end_time;

  if (OMZTensorSplitDebug)
    start_time = clock();

  // Verify that e4, e3, e1 do not exceed the maximum dimension size. Thus, we
  // will split e2 safely.
  OrigShape origShapeOfA;
  getOrigShape(inputA, &origShapeOfA);
  uint32_t maxDimSize = zdnn_get_nnpa_max_dim_idx_size();
  if ((origShapeOfA.e4 > maxDimSize) || (origShapeOfA.e3 > maxDimSize) ||
      (origShapeOfA.e1 > maxDimSize)) {
    printf("[MatMul] The 1st tensor dimension exceeds maximum dimension index "
           "size (MDIS) of %d: e4 = %d, e3 = %d, e1 = %d.\n",
        maxDimSize, origShapeOfA.e4, origShapeOfA.e3, origShapeOfA.e1);
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

  bool isSplitA = initSplitInfo(&splitInfoA);
  bool isSplitB = initSplitInfo(&splitInfoB);
  bool isSplitC = initSplitInfo(&splitInfoC);
  bool isSplitY = initSplitInfo(&splitInfoY);

  // Dim is small or ztensor split is disabled.
  if (!OMZTensorSplitEnabled || !(isSplitA || isSplitB)) {
    if (OMZTensorSplitDebug)
      printf("[MatMul] Not split zTensor ...\n");
    return call_zdnn_matmul_op(inputA, inputB, inputC, opType, output, isBcast);
  }

  // Split input A into chunks.
  if (isSplitA) {
    if (OMZTensorSplitDebug)
      printf("[MatMul] Split the 1st ztensor (A) along e2 into %d chunks of %d "
             "elements \n",
          splitInfoA.numOfChunks, splitInfoA.chunkSize);
    splitZTensor(&splitInfoA, /*copyData=*/true);
    splitZTensor(&splitInfoY, /*copyData=*/false);
  }
  // Split input B and C into chunks.
  if (isSplitB) {
    if (OMZTensorSplitDebug) {
      printf("[MatMul] Split the 2nd ztensor (B) along e1 into %d chunks of %d "
             "elements \n",
          splitInfoB.numOfChunks, splitInfoB.chunkSize);
      printf("[MatMul] Split the 3rd ztensor (C) along e1 into %d chunks of %d "
             "elements \n",
          splitInfoC.numOfChunks, splitInfoC.chunkSize);
    }
    splitZTensor(&splitInfoB, /*copyData=*/true);
    splitZTensor(&splitInfoC, /*copyData=*/true);
  }

  // Call zdnn_matmul_op on each chunk.
  // Iterate over the chunks along the first dim of A.
  for (uint32_t i = 0; i < splitInfoA.numOfChunks; ++i) {
    zdnn_ztensor *zaTensor =
        (isSplitA) ? (splitInfoA.chunks + i)->ztensor : inputA;
    zdnn_ztensor *zyTensor =
        (isSplitY) ? (splitInfoY.chunks + i)->ztensor : output;
    if (isSplitB) {
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
    } else {
      // Only A is splitted.
      zdnn_status status = call_zdnn_matmul_op(
          zaTensor, inputB, inputC, opType, zyTensor, isBcast);
      assert(status == ZDNN_OK);
    }
  }

  // Merging the chunks into the output.
  if (isSplitA) {
    mergeZTensors(&splitInfoY);
    freeSplitInfoBuffer(&splitInfoA);
    freeSplitInfoBuffer(&splitInfoY);
  }
  if (isSplitB) {
    freeSplitInfoBuffer(&splitInfoB);
    freeSplitInfoBuffer(&splitInfoC);
  }

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
