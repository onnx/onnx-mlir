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
  ZDNN_ADD_EXT,
  ZDNN_DIV_EXT,
  ZDNN_MAX_EXT,
  ZDNN_MIN_EXT,
  ZDNN_MUL_EXT,
  ZDNN_SUB_EXT,
  // Unary
  ZDNN_EXP_EXT,
  ZDNN_LOG_EXT,
  ZDNN_RELU_EXT,
  ZDNN_TANH_EXT,
  ZDNN_SIGMOID_EXT,
} ElemementwiseOp;

static zdnn_status zdnn_unary_elementwise_common(const zdnn_ztensor *input,
    const void *clippingValue, zdnn_ztensor *output, ElemementwiseOp opType) {
  // Verify that e4, e3, e1 do not exceed the maximum dimension size. Thus, we
  // will split e2 safely.
  OrigShape origShapeOfX;
  getOrigShape(input, &origShapeOfX);
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

  // Dim is small or ztensor split is disabled.
  if (!OMZTensorSplitEnabled || !initSplitInfo(&splitInfoX) ||
      !initSplitInfo(&splitInfoY)) {
    if (OMZTensorSplitDebug)
      printf("[UnaryElementwise] Not split zTensor ...\n");
    if (opType == ZDNN_EXP_EXT)
      return zdnn_exp(input, output);
    else if (opType == ZDNN_LOG_EXT)
      return zdnn_log(input, output);
    else if (opType == ZDNN_RELU_EXT)
      return zdnn_relu(input, clippingValue, output);
    else if (opType == ZDNN_SIGMOID_EXT)
      return zdnn_sigmoid(input, output);
    else if (opType == ZDNN_TANH_EXT)
      return zdnn_tanh(input, output);
    else
      return ZDNN_UNAVAILABLE_FUNCTION;
  }

  // Split input.
  if (OMZTensorSplitDebug)
    printf("[UnaryElementwise] Split the input ztensor along e2 into %d chunks "
           "of %d elements \n",
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
  // Verify that e4, e3, e1 do not exceed the maximum dimension size. Thus, we
  // will split e2 safely.
  OrigShape origShapeOfA, origShapeOfB;
  getOrigShape(inputA, &origShapeOfA);
  getOrigShape(inputB, &origShapeOfB);
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

  // Dim is small or ztensor split is disabled.
  if (!OMZTensorSplitEnabled || !initSplitInfo(&splitInfoA) ||
      !initSplitInfo(&splitInfoB) || !initSplitInfo(&splitInfoY)) {
    if (OMZTensorSplitDebug)
      printf("[BinaryElementwise] Not split zTensor ...\n");
    if (opType == ZDNN_ADD_EXT)
      return zdnn_add(inputA, inputB, output);
    else if (opType == ZDNN_SUB_EXT)
      return zdnn_sub(inputA, inputB, output);
    else if (opType == ZDNN_MUL_EXT)
      return zdnn_mul(inputA, inputB, output);
    else if (opType == ZDNN_DIV_EXT)
      return zdnn_div(inputA, inputB, output);
    else if (opType == ZDNN_MAX_EXT)
      return zdnn_max(inputA, inputB, output);
    else if (opType == ZDNN_MIN_EXT)
      return zdnn_min(inputA, inputB, output);
    else
      return ZDNN_UNAVAILABLE_FUNCTION;
  }

  // Split input.
  if (OMZTensorSplitDebug)
    printf(
        "[BinaryElementwise] Split the input ztensors along e2 into %d chunks "
        "of %d elements \n",
        splitInfoA.numOfChunks, splitInfoA.chunkSize);

  double splitTime = 0.;
  double mmTime = 0.;
  double mergeTime = 0.;
  clock_t start_time, end_time;

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
