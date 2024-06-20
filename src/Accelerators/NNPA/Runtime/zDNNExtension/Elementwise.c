/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ Elementwise.c -------------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// A wrapper of zdnn elementwise ops for ztensor partition and parallelism.
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

typedef enum ElemementwiseOp {
  // Binary
  ZDNN_ADD_EXT = 0,
  ZDNN_DIV_EXT = 1,
  ZDNN_MAX_EXT = 2,
  ZDNN_MIN_EXT = 3,
  ZDNN_MUL_EXT = 4,
  ZDNN_SUB_EXT = 5,
  // Unary
  ZDNN_EXP_EXT = 50,
  ZDNN_LOG_EXT = 51,
  ZDNN_RELU_EXT = 52,
  ZDNN_TANH_EXT = 53,
  ZDNN_SIGMOID_EXT = 54,
} ElemementwiseOp;

static SplitAxis selectSplitAxis(const zdnn_ztensor *t) {
  // We prefer to split E1 over E2 if E1 >= E2, because we can reuse the full
  // buffer in case of E1.
  UnmappedShape unmappedShape;
  getUnmappedShape(t, &unmappedShape);
  if (unmappedShape.e1 >= unmappedShape.e2)
    return E1;
  return E2;
}

static zdnn_status zdnn_unary_elementwise_common(const zdnn_ztensor *input,
    const void *clippingValue, zdnn_ztensor *output, ElemementwiseOp opType) {
  double splitTime = 0., computeTime = 0., mergeTime = 0.;
  clock_t start_time = 0, end_time = 0;

  if (OMZTensorSplitDebug)
    printf("[UnaryElementwise opType %d]\n", opType);

  // We split e1 or e2 in (e4, e3, e2, e1).
  SplitAxis axis = selectSplitAxis(input);
  uint32_t splitSize = OMZTensorSplitSize;
  SplitInfo siX, siY;
  initSplitInfo(&siX, input, axis, splitSize, /*allocTileBuffers=*/true,
      "UnaryElementwise X");
  initSplitInfo(&siY, output, axis, splitSize, /*allocTileBuffers=*/true,
      "UnaryElementwise Y");

  // Copy data from input to tiles.
  if (OMZTensorSplitDebug)
    start_time = clock();
  copyData(&siX, FULL_TO_TILES);
  if (OMZTensorSplitDebug) {
    end_time = clock();
    splitTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Call zdnn op on each tile.
  if (OMZTensorSplitDebug)
    start_time = clock();
  for (uint32_t i = 0; i < getNumOfTiles(&siX); ++i) {
    zdnn_ztensor *zx = getTile(&siX, i);
    zdnn_ztensor *zy = getTile(&siY, i);
    zdnn_status status;
    if (opType == ZDNN_EXP_EXT)
      status = zdnn_exp(zx, zy);
    else if (opType == ZDNN_LOG_EXT)
      status = zdnn_log(zx, zy);
    else if (opType == ZDNN_RELU_EXT)
      status = zdnn_relu(zx, clippingValue, zy);
    else if (opType == ZDNN_SIGMOID_EXT)
      status = zdnn_sigmoid(zx, zy);
    else if (opType == ZDNN_TANH_EXT)
      status = zdnn_tanh(zx, zy);
    else
      status = ZDNN_UNAVAILABLE_FUNCTION;
    if (status != ZDNN_OK)
      return status;
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
    printf(
        "[UnaryElementwise] split, %f, compute, %f, merge, %f (milliseconds)\n",
        splitTime, computeTime, mergeTime);

  return ZDNN_OK;
}

static zdnn_status zdnn_binary_elementwise_common(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, zdnn_ztensor *output, ElemementwiseOp opType) {
  double splitTime = 0., computeTime = 0., mergeTime = 0.;
  clock_t start_time = 0, end_time = 0;

  if (OMZTensorSplitDebug)
    printf("[BinaryElementwise opType %d]\n", opType);

  // We split e1 or e2 in (e4, e3, e2, e1).
  SplitAxis axis = selectSplitAxis(inputA);
  uint32_t splitSize = OMZTensorSplitSize;
  SplitInfo siA, siB, siY;
  initSplitInfo(&siA, inputA, axis, splitSize, /*allocTileBuffers=*/true,
      "BinaryElementwise A");
  initSplitInfo(&siB, inputB, axis, splitSize, /*allocTileBuffers=*/true,
      "BinaryElementwise B");
  initSplitInfo(&siY, output, axis, splitSize, /*allocTileBuffers=*/true,
      "BinaryElementwise Y");

  // Copy data from inputs into tiles.
  if (OMZTensorSplitDebug)
    start_time = clock();
  copyData(&siA, FULL_TO_TILES);
  copyData(&siB, FULL_TO_TILES);
  if (OMZTensorSplitDebug) {
    end_time = clock();
    splitTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Call zdnn op on each tile.
  if (OMZTensorSplitDebug)
    start_time = clock();
  for (uint32_t i = 0; i < getNumOfTiles(&siA); ++i) {
    zdnn_ztensor *za = getTile(&siA, i);
    zdnn_ztensor *zb = getTile(&siB, i);
    zdnn_ztensor *zy = getTile(&siY, i);
    zdnn_status status;
    if (opType == ZDNN_ADD_EXT)
      status = zdnn_add(za, zb, zy);
    else if (opType == ZDNN_SUB_EXT)
      status = zdnn_sub(za, zb, zy);
    else if (opType == ZDNN_MUL_EXT)
      status = zdnn_mul(za, zb, zy);
    else if (opType == ZDNN_DIV_EXT)
      status = zdnn_div(za, zb, zy);
    else if (opType == ZDNN_MAX_EXT)
      status = zdnn_max(za, zb, zy);
    else if (opType == ZDNN_MIN_EXT)
      status = zdnn_min(za, zb, zy);
    else
      status = ZDNN_UNAVAILABLE_FUNCTION;
    if (status != ZDNN_OK)
      return status;
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

  freeSplitInfoData(&siA);
  freeSplitInfoData(&siB);
  freeSplitInfoData(&siY);

  if (OMZTensorSplitDebug)
    printf("[BinaryElementwise] split, %f, compute, %f, merge, %f "
           "(milliseconds)\n",
        splitTime, computeTime, mergeTime);

  return ZDNN_OK;
}

// -----------------------------------------------------------------------------
// Extension Functions
// Same name as zdnn functions but with the `_ext` postfix.
// -----------------------------------------------------------------------------

zdnn_status zdnn_add_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_ADD_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_add");
  return status;
}

zdnn_status zdnn_sub_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_SUB_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_sub");
  return status;
}

zdnn_status zdnn_mul_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_MUL_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_mul");
  return status;
}

zdnn_status zdnn_div_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_DIV_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_div");
  return status;
}

zdnn_status zdnn_min_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_MIN_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_min");
  return status;
}

zdnn_status zdnn_max_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_MAX_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_max");
  return status;
}

zdnn_status zdnn_exp_ext(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      zdnn_unary_elementwise_common(input, NULL, output, ZDNN_EXP_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_exp");
  return status;
}

zdnn_status zdnn_log_ext(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      zdnn_unary_elementwise_common(input, NULL, output, ZDNN_LOG_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_log");
  return status;
}

zdnn_status zdnn_relu_ext(const zdnn_ztensor *input, const void *clippingValue,
    zdnn_ztensor *output) {
  zdnn_status status = zdnn_unary_elementwise_common(
      input, clippingValue, output, ZDNN_RELU_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_relu");
  return status;
}

zdnn_status zdnn_sigmoid_ext(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      zdnn_unary_elementwise_common(input, NULL, output, ZDNN_SIGMOID_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_sigmoid");
  return status;
}

zdnn_status zdnn_tanh_ext(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      zdnn_unary_elementwise_common(input, NULL, output, ZDNN_TANH_EXT);
  CHECK_ZDNN_STATUS(status, "zdnn_tanh");
  return status;
}

#ifdef __cplusplus
}
#endif
