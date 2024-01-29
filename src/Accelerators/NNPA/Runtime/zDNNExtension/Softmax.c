/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Softmax.c ---------------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// A wrapper of zdnn_softmax for ztensor partition and parallelism.
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

zdnn_status zdnn_softmax_ext(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output) {
  // Verify that e4, e3, e1 do not exceed the maximum dimension size. Thus, we
  // will split e2 safely.
  OrigShape origShapeOfX;
  getOrigShape(input, &origShapeOfX);
  uint32_t maxDimSize = zdnn_get_nnpa_max_dim_idx_size();
  if ((origShapeOfX.e4 > maxDimSize) || (origShapeOfX.e3 > maxDimSize) ||
      (origShapeOfX.e1 > maxDimSize)) {
    printf(
        "[Softmax] The input tensor dimension exceeds maximum dimension index "
        "size (MDIS) of %d: e4 = %d, e3 = %d, e1 = %d.\n",
        maxDimSize, origShapeOfX.e4, origShapeOfX.e3, origShapeOfX.e1);
    return ZDNN_EXCEEDS_MDIS;
  }

  // We split e2 in (e4, e3, e2, e1).
  SplitInfo splitInfoX = {
      .origZTensor = input, .axis = 2, .chunkSize = OMZTensorSplitSize};
  SplitInfo splitInfoY = {
      .origZTensor = output, .axis = 2, .chunkSize = OMZTensorSplitSize};

  // Dim is small or ztensor split is disabled.
  if (!OMZTensorSplitEnabled ||
      !(initSplitInfo(&splitInfoX) && initSplitInfo(&splitInfoY))) {
    if (OMZTensorSplitDebug)
      printf("[Softmax] Not split zTensor ...\n");
    return zdnn_softmax(input, save_area, act_func, output);
  }

  // Split input.
  if (OMZTensorSplitDebug)
    printf("[Softmax] Split the input ztensor along e2 into %d chunks of %d "
           "elements \n",
        splitInfoX.numOfChunks, splitInfoX.chunkSize);

  double splitTime = 0.;
  double mmTime = 0.;
  double mergeTime = 0.;
  clock_t start_time, end_time;

  // Split input into chunks.
  if (OMZTensorSplitDebug)
    start_time = clock();
  splitZTensor(&splitInfoX, /*copyData=*/true);
  splitZTensor(&splitInfoY, /*copyData=*/false);
  if (OMZTensorSplitDebug) {
    end_time = clock();
    splitTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Call zdnn_matmul_op on each chunk. Not use save_area.
  // TODO: could we reuse save_area in particular in the parallel scenario?
  if (OMZTensorSplitDebug)
    start_time = clock();
  for (uint32_t i = 0; i < splitInfoX.numOfChunks; ++i) {
    zdnn_ztensor *zxTensor = (splitInfoX.chunks + i)->ztensor;
    zdnn_ztensor *zyTensor = (splitInfoY.chunks + i)->ztensor;
    zdnn_status status = zdnn_softmax(zxTensor, NULL, act_func, zyTensor);
    assert(status == ZDNN_OK);
  }
  if (OMZTensorSplitDebug) {
    end_time = clock();
    mmTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Merging the chunks into the output.
  if (OMZTensorSplitDebug)
    start_time = clock();
  mergeZTensors(&splitInfoY);
  if (OMZTensorSplitDebug) {
    end_time = clock();
    mergeTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  freeSplitInfoBuffer(&splitInfoX);
  freeSplitInfoBuffer(&splitInfoY);

  if (OMZTensorSplitDebug)
    printf("[Softmax] split, %f, mm, %f, merge, %f (milliseconds)\n", splitTime,
        mmTime, mergeTime);

  return ZDNN_OK;
}

#ifdef __cplusplus
}
#endif
