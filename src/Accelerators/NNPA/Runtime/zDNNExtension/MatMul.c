/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- MatMul.c ----------------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// A wrapper of zdnn_matmul_op supports very large input tensors.
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

#include "src/Accelerators/NNPA/Runtime/zDNNExtension/zDNNExtension.h"
#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

zdnn_status call_zdnn_matmul_op(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output, bool isBcast) {
  zdnn_status status;
  if (isBcast) {
    status = zdnn_matmul_bcast_op(
        inputA, inputB, inputC, (zdnn_matmul_bcast_ops)opType, output);
  } else {
    status =
        zdnn_matmul_op(inputA, inputB, inputC, (zdnn_matmul_ops)opType, output);
  }
  return status;
}

zdnn_status zdnn_matmul_op_common(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output, bool isBcast) {
  zdnn_status status;
  bool isDebug = ZTensorSplitDebugFromEnv();

  // For a MatMul of (M,N)*(N,P),
  // We split M that is the third one in the original 4D tensor.
  uint32_t axis = 2;
  uint32_t chunkSize = ZTensorSplitSizeFromEnv();
  SplitInfo splitInfoA, splitInfoY;

  // Dim is small or ztensor split is disabled.
  if (!ZTensorSplitEnabledFromEnv() ||
      !initSplitInfo(inputA, axis, chunkSize, &splitInfoA)) {
    if (isDebug)
      printf("Not split zTensor ...\n");
    status =
        call_zdnn_matmul_op(inputA, inputB, inputC, opType, output, isBcast);
    assert(status == ZDNN_OK && "Failed to call zdnn_matmul_op");
    return status;
  }

  // Split input A.
  if (isDebug)
    printf("Split zTensor A along axis %d into %d chunks of %d elements \n",
        axis, splitInfoA.numOfChunks, splitInfoA.chunkSize);
  initSplitInfo(output, axis, chunkSize, &splitInfoY);

  double splitTime = 0.;
  double mmTime = 0.;
  double mergeTime = 0.;
  clock_t start_time, end_time;

  // Split input A along the second innermost axis into chunks.
  if (isDebug)
    start_time = clock();
  splitZTensor(inputA, &splitInfoA, /*copyData=*/true);
  splitZTensor(output, &splitInfoY, /*copyData=*/false);
  if (isDebug) {
    end_time = clock();
    splitTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Call zdnn_matmul_op on each chunk.
  if (isDebug)
    start_time = clock();
  for (uint32_t i = 0; i < splitInfoA.numOfChunks; ++i) {
    status = call_zdnn_matmul_op(splitInfoA.tensors + i, inputB, inputC, opType,
        splitInfoY.tensors + i, isBcast);
    assert(status == ZDNN_OK);
  }
  if (isDebug) {
    end_time = clock();
    mmTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  // Merging the chunks into the output.
  if (isDebug)
    start_time = clock();
  mergeZTensors(&splitInfoY, output);
  if (isDebug) {
    end_time = clock();
    mergeTime = ((float)(end_time - start_time) / (float)CLOCKS_PER_SEC) * 1000;
  }

  freeSplitInfoBuffer(&splitInfoA);
  freeSplitInfoBuffer(&splitInfoY);

  if (isDebug)
    printf("split, %f, mm, %f, merge, %f (milliseconds)\n", splitTime, mmTime,
        mergeTime);

  assert(status == ZDNN_OK && "Failed to call zdnn_matmul_op");
  return status;
}

zdnn_status zdnn_matmul_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output) {
  return zdnn_matmul_op_common(
      inputA, inputB, inputC, opType, output, /*isBcast=*/false);
}

zdnn_status zdnn_matmul_bcast_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output) {
  return zdnn_matmul_op_common(
      inputA, inputB, inputC, opType, output, /*isBcast=*/true);
}

#ifdef __cplusplus
}
#endif
