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

  // For a MatMul of A(M,N)*B(N,P)+C(P),
  // We split M that is e2 in (e4, e3, e2, e1), and P that is e1.
  SplitInfo splitInfoA = {.fullZTensor = inputA,
      .axis = E2,
      .numOfElemsPerTile = OMZTensorSplitSize};
  SplitInfo splitInfoB = {.fullZTensor = inputB,
      .axis = E1,
      .numOfElemsPerTile = OMZTensorSplitSize};
  SplitInfo splitInfoC = {.fullZTensor = inputC,
      .axis = E1,
      .numOfElemsPerTile = OMZTensorSplitSize};
  SplitInfo splitInfoY = {.fullZTensor = output,
      .axis = E2,
      .numOfElemsPerTile = OMZTensorSplitSize};

  initSplitInfo(&splitInfoA, /*allocTileBuffers=*/true, "MatMul A");
  initSplitInfo(&splitInfoB, /*allocTileBuffers=*/true, "MatMul B");
  initSplitInfo(&splitInfoC, /*allocTileBuffers=*/true, "MatMul C");
  initSplitInfo(&splitInfoY, /*allocTileBuffers=*/true, "MatMul Y");

  // Copy data from A, B, C into their tiles.
  copyData(&splitInfoA, FULL_TO_TILES);
  copyData(&splitInfoB, FULL_TO_TILES);
  copyData(&splitInfoC, FULL_TO_TILES);

  // Call zdnn_matmul_op on each tile.
  // Iterate over the tiles along the first dim of A.
  for (uint32_t i = 0; i < getNumOfTiles(&splitInfoA); ++i) {
    zdnn_ztensor *zaTensor = getTile(&splitInfoA, i);
    zdnn_ztensor *zyTensor = getTile(&splitInfoY, i);

    SplitInfo splitInfoYB = {.fullZTensor = zyTensor,
        .axis = E1,
        .numOfElemsPerTile = OMZTensorSplitSize};
    initSplitInfo(&splitInfoYB, /*allocTileBuffers=*/true, "MatMul YB");
    // Iterate over the tiles along the second dim of B.
    for (uint32_t j = 0; j < getNumOfTiles(&splitInfoB); ++j) {
      zdnn_ztensor *zbTensor = getTile(&splitInfoB, j);
      zdnn_ztensor *zcTensor = getTile(&splitInfoC, j);
      zdnn_ztensor *zybTensor = getTile(&splitInfoYB, j);
      zdnn_status status = call_zdnn_matmul_op(
          zaTensor, zbTensor, zcTensor, opType, zybTensor, isBcast);
      assert(status == ZDNN_OK);
    }
    copyData(&splitInfoYB, TILES_TO_FULL);
    freeSplitInfoData(&splitInfoYB);
  }

  // Copy data from the tiles back to the full ztensor.
  copyData(&splitInfoY, TILES_TO_FULL);

  // Free temporary buffers.
  freeSplitInfoData(&splitInfoA);
  freeSplitInfoData(&splitInfoB);
  freeSplitInfoData(&splitInfoC);
  freeSplitInfoData(&splitInfoY);

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
