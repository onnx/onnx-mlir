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

// -----------------------------------------------------------------------------
// Extension Functions
// Same name as zdnn functions but with the `_ext` postfix.
// -----------------------------------------------------------------------------

zdnn_status zdnn_softmax_ext(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output) {
  double splitTime = 0., computeTime = 0., mergeTime = 0.;
  clock_t start_time = 0, end_time = 0;

  // Verify that e3, e2, e1 do not exceed the maximum dimension size. Thus, we
  // will split e4 safely.
  UnmappedShape unmappedShapeOfX;
  getUnmappedShape(input, &unmappedShapeOfX);
  if (OMZTensorSplitDebug) {
    printf("[Softmax] X:  e4 = %d, e3 = %d, e2 = %d, e1 = %d.\n",
        unmappedShapeOfX.e4, unmappedShapeOfX.e3, unmappedShapeOfX.e2,
        unmappedShapeOfX.e1);
  }
  uint32_t maxDimSize = zdnn_get_nnpa_max_dim_idx_size();
  if ((unmappedShapeOfX.e3 > maxDimSize) ||
      (unmappedShapeOfX.e2 > maxDimSize) ||
      (unmappedShapeOfX.e1 > maxDimSize)) {
    printf(
        "[Softmax] The input tensor dimension exceeds maximum dimension index "
        "size (MDIS) of %d: e3 = %d, e2 = %d, e1 = %d.\n",
        maxDimSize, unmappedShapeOfX.e3, unmappedShapeOfX.e2,
        unmappedShapeOfX.e1);
    return ZDNN_EXCEEDS_MDIS;
  }

  // We split e4 in (e4, e3, e2, e1) to reuse the orignal buffer.
  SplitInfo splitInfoX = {.fullZTensor = input,
      .axis = E4,
      .numOfElemsPerTile = OMZTensorSplitSize};
  SplitInfo splitInfoY = {.fullZTensor = output,
      .axis = E4,
      .numOfElemsPerTile = OMZTensorSplitSize};
  initSplitInfo(&splitInfoX);
  initSplitInfo(&splitInfoY);

  if (OMZTensorSplitDebug)
    printf("[Softmax] Split the input ztensor along e4 into %d tiles of %d "
           "elements. ReuseZTensor: %d, ReuseBuffer: %d \n",
        splitInfoX.numOfTiles, splitInfoX.numOfElemsPerTile,
        splitInfoX.reuseFullZTensor, splitInfoX.reuseFullBuffer);

  // Copy data from input to tiles.
  if (OMZTensorSplitDebug)
    start_time = clock();
  copyData(&splitInfoX, FULL_TO_TILES);
  if (OMZTensorSplitDebug) {
    end_time = clock();
    splitTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Call zdnn_softmax on each tile. Not use save_area.
  // TODO: could we reuse save_area in particular in the parallel scenario?
  if (OMZTensorSplitDebug)
    start_time = clock();
  for (uint32_t i = 0; i < splitInfoX.numOfTiles; ++i) {
    zdnn_ztensor *zxTensor = splitInfoX.tiles + i;
    zdnn_ztensor *zyTensor = splitInfoY.tiles + i;
    zdnn_status status = zdnn_softmax(zxTensor,
        (splitInfoX.reuseFullZTensor) ? save_area : NULL, act_func, zyTensor);
    assert(status == ZDNN_OK);
  }
  if (OMZTensorSplitDebug) {
    end_time = clock();
    computeTime =
        ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Copy data from tiles to the output.
  if (OMZTensorSplitDebug)
    start_time = clock();
  copyData(&splitInfoY, TILES_TO_FULL);
  if (OMZTensorSplitDebug) {
    end_time = clock();
    mergeTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  freeSplitInfoBuffer(&splitInfoX);
  freeSplitInfoBuffer(&splitInfoY);

  if (OMZTensorSplitDebug)
    printf("[Softmax] split, %f, compute, %f, merge, %f (milliseconds)\n",
        splitTime, computeTime, mergeTime);

  return ZDNN_OK;
}

#ifdef __cplusplus
}
#endif
