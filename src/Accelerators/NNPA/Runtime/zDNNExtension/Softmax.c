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

// z/OS specific includes
#ifdef __MVS__
// needed for pthread on z/OS
#define _OPEN_THREADS
// z/OS needs <time.h> in addition to <sys/time.h>
#include <time.h>
#endif

#include <assert.h>
#include <math.h>
#include <pthread.h>
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

  // We split e4 in (e4, e3, e2, e1) to reuse the full buffer.
  uint32_t splitSize = OMZTensorSplitSize;
  SplitInfo siX, siY;
  initSplitInfo(
      &siX, input, E4, splitSize, /*allocTileBuffers=*/true, "Softmax X");
  initSplitInfo(
      &siY, output, E4, splitSize, /*allocTileBuffers=*/true, "Softmax Y");

  // Copy data from input to tiles.
  if (OMZTensorSplitDebug)
    start_time = clock();
  copyData(&siX, FULL_TO_TILES);
  if (OMZTensorSplitDebug) {
    end_time = clock();
    splitTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Call zdnn_softmax on each tile. Not use save_area.
  // TODO: could we reuse save_area in particular in the parallel scenario?
  if (OMZTensorSplitDebug)
    start_time = clock();
  for (uint32_t i = 0; i < getNumOfTiles(&siX); ++i) {
    zdnn_ztensor *zx = getTile(&siX, i);
    zdnn_ztensor *zy = getTile(&siY, i);
    zdnn_status status = zdnn_softmax(
        zx, (siX.reuseFullZTensor) ? save_area : NULL, act_func, zy);
    CHECK_ZDNN_STATUS(status, "zdnn_softmax");
  }
  if (OMZTensorSplitDebug) {
    end_time = clock();
    computeTime =
        ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Copy data from tiles to the output.
  if (OMZTensorSplitDebug)
    start_time = clock();
  copyData(&siY, TILES_TO_FULL);
  if (OMZTensorSplitDebug) {
    end_time = clock();
    mergeTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  freeSplitInfoData(&siX);
  freeSplitInfoData(&siY);

  if (OMZTensorSplitDebug)
    printf("[Softmax] split, %f, compute, %f, merge, %f (milliseconds)\n",
        splitTime, computeTime, mergeTime);

  return ZDNN_OK;
}

#ifdef __cplusplus
}
#endif
