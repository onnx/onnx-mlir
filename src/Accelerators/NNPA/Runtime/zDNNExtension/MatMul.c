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
#include <sched.h>
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

static float get_elapse(const struct timeval start_t, const struct timeval end_t) {
  return 
  (((end_t.tv_sec * 1000000.) + end_t.tv_usec) - ((start_t.tv_sec * 1000000) + start_t.tv_usec))/1000;
}

// It is supposed that sched.h should have the declaration of sched_getcpu.
// No problem when I compiled a standalone test case.
// But in onnx-mlir, this function is not defined.
// Explicitly define it here
extern int sched_getcpu();

static zdnn_status zdnn_matmul_op_common(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output, bool isBcast) {
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

  // For a MatMul of (M,N)*(N,P),
  // We split M that is e2 in (e4, e3, e2, e1).
  SplitInfo splitInfoA = {
      .origZTensor = inputA, .axis = 2, .chunkSize = OMZTensorSplitSize};
  SplitInfo splitInfoY = {
      .origZTensor = output, .axis = 2, .chunkSize = OMZTensorSplitSize};

  double splitTime = 0.;
  double mmTime = 0.;
  double mergeTime = 0.;
  struct timeval start_t, end_t;
  float elapse;

  // Dim is small or ztensor split is disabled.
  if (!OMZTensorSplitEnabled || !initSplitInfo(&splitInfoA) ||
      !initSplitInfo(&splitInfoY)) {
    if (OMZTensorSplitDebug)
      printf("[MatMul] Not split zTensor ...\n");
    if (OMZTensorSplitDebug)
      gettimeofday(&start_t, NULL);
    zdnn_status status = call_zdnn_matmul_op(inputA, inputB, inputC, opType, output, isBcast);
    assert(status == ZDNN_OK && ("call_zdnn_matmul_op failed"));
  if (OMZTensorSplitDebug) {
    gettimeofday(&end_t, NULL);
    elapse = get_elapse(start_t, end_t);
    printf("[MatMul]  mm, %f, (milliseconds)\n", elapse);
  }
    return status;
  }

  // Split input A.
  if (OMZTensorSplitDebug)
    printf("[MatMul] Split the 1st ztensor along e2 into %d chunks of %d "
           "elements \n",
        splitInfoA.numOfChunks, splitInfoA.chunkSize);

  // Split input A into chunks.
  if (OMZTensorSplitDebug)
    gettimeofday(&start_t, NULL);
  splitZTensor(&splitInfoA, /*copyData=*/true);
  splitZTensor(&splitInfoY, /*copyData=*/false);
  if (OMZTensorSplitDebug) {
    gettimeofday(&end_t, NULL);
    splitTime = get_elapse(start_t, end_t);
  }

  // Call zdnn_matmul_op on each chunk.
  if (OMZTensorSplitDebug)
    gettimeofday(&start_t, NULL);

  // Parallelize the mm part over each chunk
  // Thread binding is done at runtime with OMP_PLACES and OMP_PROC_BIND
#pragma omp parallel for proc_bind(spread)
  for (uint32_t i = 0; i < splitInfoA.numOfChunks; ++i) {
    zdnn_ztensor *zaTensor = (splitInfoA.chunks + i)->ztensor;
    zdnn_ztensor *zyTensor = (splitInfoY.chunks + i)->ztensor;
    zdnn_status status = call_zdnn_matmul_op(
        zaTensor, inputB, inputC, opType, zyTensor, isBcast);
    assert(status == ZDNN_OK);
    if (OMZTensorSplitDebug) {
      printf("====omp thread %u) is on cpu %d=======\n", i, sched_getcpu());
    }
  }
  if (OMZTensorSplitDebug) {
    gettimeofday(&end_t, NULL);
    mmTime = get_elapse(start_t, end_t);
  }

  // Merging the chunks into the output.
  if (OMZTensorSplitDebug)
    gettimeofday(&start_t, NULL);
  mergeZTensors(&splitInfoY);
  if (OMZTensorSplitDebug) {
    gettimeofday(&end_t, NULL);
    mergeTime = get_elapse(start_t, end_t);
  }

  freeSplitInfoBuffer(&splitInfoA);
  freeSplitInfoBuffer(&splitInfoY);

  if (OMZTensorSplitDebug)
    printf("[MatMul] split, %f, mm, %f, merge, %f (milliseconds)\n", splitTime,
        mmTime, mergeTime);

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
  assert(status == ZDNN_OK);
  return status;
}

#ifdef __cplusplus
}
#endif
