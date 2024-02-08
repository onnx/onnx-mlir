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
  UnmappedShape unmappedShapeOfA, unmappedShapeOfB, unmappedShapeOfC;
  getUnmappedShape(inputA, &unmappedShapeOfA);
  getUnmappedShape(inputB, &unmappedShapeOfB);
  getUnmappedShape(inputC, &unmappedShapeOfC);
  if (OMZTensorSplitDebug) {
    printf("[MatMul] A:  e4 = %d, e3 = %d, e2 = %d, e1 = %d.\n",
        unmappedShapeOfA.e4, unmappedShapeOfA.e3, unmappedShapeOfA.e2,
        unmappedShapeOfA.e1);
    printf("[MatMul] B:  e4 = %d, e3 = %d, e2 = %d, e1 = %d.\n",
        unmappedShapeOfA.e4, unmappedShapeOfB.e3, unmappedShapeOfB.e2,
        unmappedShapeOfB.e1);
    printf("[MatMul] C:  e4 = %d, e3 = %d, e2 = %d, e1 = %d.\n",
        unmappedShapeOfA.e4, unmappedShapeOfC.e3, unmappedShapeOfC.e2,
        unmappedShapeOfC.e1);
  }
  uint32_t maxDimSize = zdnn_get_nnpa_max_dim_idx_size();
  if ((unmappedShapeOfA.e4 > maxDimSize) ||
      (unmappedShapeOfA.e3 > maxDimSize)) {
    printf("[MatMul] The 1st tensor dimension exceeds maximum dimension index "
           "size (MDIS) of %d: e4 = %d, e3 = %d.\n",
        maxDimSize, unmappedShapeOfA.e4, unmappedShapeOfA.e3);
    return ZDNN_EXCEEDS_MDIS;
  }
  if ((unmappedShapeOfB.e4 > maxDimSize) ||
      (unmappedShapeOfB.e3 > maxDimSize)) {
    printf("[MatMul] The 2nd tensor dimension exceeds maximum dimension index "
           "size (MDIS) of %d: e4 = %d, e3 = %d.\n",
        maxDimSize, unmappedShapeOfB.e4, unmappedShapeOfB.e3);
    return ZDNN_EXCEEDS_MDIS;
  }

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

  initSplitInfo(&splitInfoA);
  initSplitInfo(&splitInfoB);
  initSplitInfo(&splitInfoC);
  initSplitInfo(&splitInfoY);

  if (OMZTensorSplitDebug) {
    printf("[MatMul] Split the 1st ztensor (A) along e2 into %d tiles of %d "
           "elements. ReuseZTensor: %d, ReuseBuffer: %d \n",
        splitInfoA.numOfTiles, splitInfoA.numOfElemsPerTile,
        splitInfoA.reuseFullZTensor, splitInfoA.reuseFullBuffer);
    printf("[MatMul] Split the 2nd ztensor (B) along e1 into %d tiles of %d "
           "elements. ReuseZTensor: %d, ReuseBuffer: %d \n",
        splitInfoB.numOfTiles, splitInfoB.numOfElemsPerTile,
        splitInfoB.reuseFullZTensor, splitInfoB.reuseFullBuffer);
    printf("[MatMul] Split the 3rd ztensor (C) along e1 into %d tiles of %d "
           "elements. ReuseZTensor: %d, ReuseBuffer: %d \n",
        splitInfoC.numOfTiles, splitInfoC.numOfElemsPerTile,
        splitInfoC.reuseFullZTensor, splitInfoC.reuseFullBuffer);
  }

  // Copy data from A, B, C into their tiles.
  copyData(&splitInfoA, FULL_TO_TILES);
  copyData(&splitInfoB, FULL_TO_TILES);
  copyData(&splitInfoC, FULL_TO_TILES);

  // Call zdnn_matmul_op on each tile.
  // Iterate over the tiles along the first dim of A.
  for (uint32_t i = 0; i < splitInfoA.numOfTiles; ++i) {
    zdnn_ztensor *zaTensor = splitInfoA.tiles + i;
    zdnn_ztensor *zyTensor = splitInfoY.tiles + i;

    SplitInfo splitInfoYB = {.fullZTensor = zyTensor,
        .axis = E1,
        .numOfElemsPerTile = OMZTensorSplitSize};
    initSplitInfo(&splitInfoYB);
    // Iterate over the tiles along the second dim of B.
    for (uint32_t j = 0; j < splitInfoB.numOfTiles; ++j) {
      zdnn_ztensor *zbTensor = splitInfoB.tiles + j;
      zdnn_ztensor *zcTensor = splitInfoC.tiles + j;
      zdnn_ztensor *zybTensor = splitInfoYB.tiles + j;
      zdnn_status status = call_zdnn_matmul_op(
          zaTensor, zbTensor, zcTensor, opType, zybTensor, isBcast);
      assert(status == ZDNN_OK);
    }
    copyData(&splitInfoYB, TILES_TO_FULL);
    freeSplitInfoBuffer(&splitInfoYB);
  }

  // Copy data from the tiles back to the full ztensor.
  copyData(&splitInfoY, TILES_TO_FULL);

  // Free temporary buffers.
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
