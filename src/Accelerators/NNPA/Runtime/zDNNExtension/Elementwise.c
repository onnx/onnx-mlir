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

static zdnn_status zdnn_unary_elementwise_common(const zdnn_ztensor *input,
    const void *clippingValue, zdnn_ztensor *output, ElemementwiseOp opType) {
  double splitTime = 0., mmTime = 0., mergeTime = 0.;
  clock_t start_time = 0, end_time = 0;

  // Verify that e4, e3, e1 do not exceed the maximum dimension size. Thus, we
  // will split e2 safely.
  OrigShape origShapeOfX;
  getOrigShape(input, &origShapeOfX);
  if (OMZTensorSplitDebug) {
    printf("[UnaryElementwise opType %d] X:  e4 = %d, e3 = %d, e2 = %d, e1 = "
           "%d.\n",
        opType, origShapeOfX.e4, origShapeOfX.e3, origShapeOfX.e2,
        origShapeOfX.e1);
  }
  uint32_t maxDimSize = zdnn_get_nnpa_max_dim_idx_size();
  if ((origShapeOfX.e4 > maxDimSize) || (origShapeOfX.e3 > maxDimSize) ||
      (origShapeOfX.e1 > maxDimSize)) {
    printf("[UnaryElementwise] The input tensor dimension exceeds maximum "
           "dimension index "
           "size (MDIS) of %d: e4 = %d, e3 = %d, e1 = %d.\n",
        maxDimSize, origShapeOfX.e4, origShapeOfX.e3, origShapeOfX.e1);
    return ZDNN_EXCEEDS_MDIS;
  }

  // We split e2 in (e4, e3, e2, e1).
  SplitInfo splitInfoX = {
      .origZTensor = input, .axis = 2, .chunkSize = OMZTensorSplitSize};
  SplitInfo splitInfoY = {
      .origZTensor = output, .axis = 2, .chunkSize = OMZTensorSplitSize};
  initSplitInfo(&splitInfoX);
  initSplitInfo(&splitInfoY);

  // Split input.
  if (OMZTensorSplitDebug)
    printf("[UnaryElementwise] Split the input ztensor along e2 into %d chunks "
           "of %d elements. ReuseZTensor: %d, ReuseBuffer: %d \n",
        splitInfoX.numOfChunks, splitInfoX.chunkSize,
        splitInfoX.reuseOrigZTensor, splitInfoX.reuseOrigBuffer);

  // Split input into chunks.
  if (OMZTensorSplitDebug)
    start_time = clock();
  splitZTensor(&splitInfoX, /*copyData=*/true);
  splitZTensor(&splitInfoY, /*copyData=*/false);
  if (OMZTensorSplitDebug) {
    end_time = clock();
    splitTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Call zdnn op on each chunk.
  if (OMZTensorSplitDebug)
    start_time = clock();
  for (uint32_t i = 0; i < splitInfoX.numOfChunks; ++i) {
    zdnn_ztensor *zxTensor = (splitInfoX.chunks + i)->ztensor;
    zdnn_ztensor *zyTensor = (splitInfoY.chunks + i)->ztensor;
    zdnn_status status;
    if (opType == ZDNN_EXP_EXT)
      status = zdnn_exp(zxTensor, zyTensor);
    else if (opType == ZDNN_LOG_EXT)
      status = zdnn_log(zxTensor, zyTensor);
    else if (opType == ZDNN_RELU_EXT)
      status = zdnn_relu(zxTensor, clippingValue, zyTensor);
    else if (opType == ZDNN_SIGMOID_EXT)
      status = zdnn_sigmoid(zxTensor, zyTensor);
    else if (opType == ZDNN_TANH_EXT)
      status = zdnn_tanh(zxTensor, zyTensor);
    else
      status = ZDNN_UNAVAILABLE_FUNCTION;
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
    printf("[UnaryElementwise] split, %f, mm, %f, merge, %f (milliseconds)\n",
        splitTime, mmTime, mergeTime);

  return ZDNN_OK;
}

static zdnn_status zdnn_binary_elementwise_common(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, zdnn_ztensor *output, ElemementwiseOp opType) {
  double splitTime = 0., mmTime = 0., mergeTime = 0.;
  clock_t start_time = 0, end_time = 0;

  // Verify that e4, e3, e1 do not exceed the maximum dimension size. Thus, we
  // will split e2 safely.
  OrigShape origShapeOfA, origShapeOfB;
  getOrigShape(inputA, &origShapeOfA);
  getOrigShape(inputB, &origShapeOfB);
  if (OMZTensorSplitDebug) {
    printf("[BinaryElementwise opType %d] A:  e4 = %d, e3 = %d, e2 = %d, e1 = "
           "%d.\n",
        opType, origShapeOfA.e4, origShapeOfA.e3, origShapeOfA.e2,
        origShapeOfA.e1);
    printf("[BinaryElementwise opType %d] B:  e4 = %d, e3 = %d, e2 = %d, e1 = "
           "%d.\n",
        opType, origShapeOfB.e4, origShapeOfB.e3, origShapeOfB.e2,
        origShapeOfB.e1);
  }
  uint32_t maxDimSize = zdnn_get_nnpa_max_dim_idx_size();
  if ((origShapeOfA.e4 > maxDimSize) || (origShapeOfA.e3 > maxDimSize) ||
      (origShapeOfA.e1 > maxDimSize)) {
    printf("[BinaryElementwise] The 1st tensor dimension exceeds maximum "
           "dimension index "
           "size (MDIS) of %d: e4 = %d, e3 = %d, e1 = %d.\n",
        maxDimSize, origShapeOfA.e4, origShapeOfA.e3, origShapeOfA.e1);
    return ZDNN_EXCEEDS_MDIS;
  }
  if ((origShapeOfB.e4 > maxDimSize) || (origShapeOfB.e3 > maxDimSize) ||
      (origShapeOfB.e1 > maxDimSize)) {
    printf("[BinaryElementwise] The 2nd tensor dimension exceeds maximum "
           "dimension index "
           "size (MDIS) of %d: e4 = %d, e3 = %d, e1 = %d.\n",
        maxDimSize, origShapeOfB.e4, origShapeOfB.e3, origShapeOfB.e1);
    return ZDNN_EXCEEDS_MDIS;
  }

  // We split e2 in (e4, e3, e2, e1).
  SplitInfo splitInfoA = {
      .origZTensor = inputA, .axis = 2, .chunkSize = OMZTensorSplitSize};
  SplitInfo splitInfoB = {
      .origZTensor = inputB, .axis = 2, .chunkSize = OMZTensorSplitSize};
  SplitInfo splitInfoY = {
      .origZTensor = output, .axis = 2, .chunkSize = OMZTensorSplitSize};
  initSplitInfo(&splitInfoA);
  initSplitInfo(&splitInfoB);
  initSplitInfo(&splitInfoY);

  // Split input.
  if (OMZTensorSplitDebug)
    printf(
        "[BinaryElementwise] Split the input ztensors along e2 into %d chunks "
        "of %d elements. ReuseZTensor: %d, ReuseBuffer: %d \n",
        splitInfoA.numOfChunks, splitInfoA.chunkSize,
        splitInfoA.reuseOrigZTensor, splitInfoA.reuseOrigBuffer);

  // Split input into chunks.
  if (OMZTensorSplitDebug)
    start_time = clock();
  splitZTensor(&splitInfoA, /*copyData=*/true);
  splitZTensor(&splitInfoB, /*copyData=*/true);
  splitZTensor(&splitInfoY, /*copyData=*/false);
  if (OMZTensorSplitDebug) {
    end_time = clock();
    splitTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Call zdnn op on each chunk.
  if (OMZTensorSplitDebug)
    start_time = clock();
  for (uint32_t i = 0; i < splitInfoA.numOfChunks; ++i) {
    zdnn_ztensor *zaTensor = (splitInfoA.chunks + i)->ztensor;
    zdnn_ztensor *zbTensor = (splitInfoB.chunks + i)->ztensor;
    zdnn_ztensor *zyTensor = (splitInfoY.chunks + i)->ztensor;
    zdnn_status status;
    if (opType == ZDNN_ADD_EXT)
      status = zdnn_add(zaTensor, zbTensor, zyTensor);
    else if (opType == ZDNN_SUB_EXT)
      status = zdnn_sub(zaTensor, zbTensor, zyTensor);
    else if (opType == ZDNN_MUL_EXT)
      status = zdnn_mul(zaTensor, zbTensor, zyTensor);
    else if (opType == ZDNN_DIV_EXT)
      status = zdnn_div(zaTensor, zbTensor, zyTensor);
    else if (opType == ZDNN_MAX_EXT)
      status = zdnn_max(zaTensor, zbTensor, zyTensor);
    else if (opType == ZDNN_MIN_EXT)
      status = zdnn_min(zaTensor, zbTensor, zyTensor);
    else
      status = ZDNN_UNAVAILABLE_FUNCTION;
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

  freeSplitInfoBuffer(&splitInfoA);
  freeSplitInfoBuffer(&splitInfoB);
  freeSplitInfoBuffer(&splitInfoY);

  if (OMZTensorSplitDebug)
    printf("[BinaryElementwise] split, %f, mm, %f, merge, %f (milliseconds)\n",
        splitTime, mmTime, mergeTime);

  return ZDNN_OK;
}

// -----------------------------------------------------------------------------
// Extension Functions
// Same name as zdnn functions but with the `_ext` postfix.
// -----------------------------------------------------------------------------

zdnn_status zdnn_add_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  return zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_ADD_EXT);
}

zdnn_status zdnn_sub_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  return zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_SUB_EXT);
}

zdnn_status zdnn_mul_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  return zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_MUL_EXT);
}

zdnn_status zdnn_div_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  return zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_DIV_EXT);
}

zdnn_status zdnn_min_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  return zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_MIN_EXT);
}

zdnn_status zdnn_max_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  return zdnn_binary_elementwise_common(inputA, inputB, output, ZDNN_MAX_EXT);
}

zdnn_status zdnn_exp_ext(const zdnn_ztensor *input, zdnn_ztensor *output) {
  return zdnn_unary_elementwise_common(input, NULL, output, ZDNN_EXP_EXT);
}

zdnn_status zdnn_log_ext(const zdnn_ztensor *input, zdnn_ztensor *output) {
  return zdnn_unary_elementwise_common(input, NULL, output, ZDNN_LOG_EXT);
}

zdnn_status zdnn_relu_ext(const zdnn_ztensor *input, const void *clippingValue,
    zdnn_ztensor *output) {
  return zdnn_unary_elementwise_common(
      input, clippingValue, output, ZDNN_RELU_EXT);
}

zdnn_status zdnn_sigmoid_ext(const zdnn_ztensor *input, zdnn_ztensor *output) {
  return zdnn_unary_elementwise_common(input, NULL, output, ZDNN_SIGMOID_EXT);
}

zdnn_status zdnn_tanh_ext(const zdnn_ztensor *input, zdnn_ztensor *output) {
  return zdnn_unary_elementwise_common(input, NULL, output, ZDNN_TANH_EXT);
}

#ifdef __cplusplus
}
#endif
