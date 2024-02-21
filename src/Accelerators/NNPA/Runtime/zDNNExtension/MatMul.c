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
#define _OPEN_SYS_EXT
#include <sys/ps.h>
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

#ifndef __MVS__
// It is supposed that sched.h should have the declaration of sched_getcpu.
// No problem when a standalone test case is compiled with clang or g++.
// But in onnx-mlir, this function is not defined. Explicitly define it here
// ToFix: find the correct include file.
extern int sched_getcpu();
#endif

static zdnn_status zdnn_matmul_op_common(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output, bool isBcast) {
  double totalTime = 0.;
  struct timeval start_t, end_t;
  struct timeval start_t1, end_t1;

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

  if (OMZTensorSplitDebug) {
    gettimeofday(&start_t, NULL);
  }

  initSplitInfo(&splitInfoA, true, "MatMul A");
  initSplitInfo(&splitInfoB, true, "MatMul B");
  initSplitInfo(&splitInfoC, true, "MatMul C");
  initSplitInfo(&splitInfoY, true, "MatMul Y");

  // Copy data from A, B, C into their tiles.
  copyData(&splitInfoA, FULL_TO_TILES);
  copyData(&splitInfoB, FULL_TO_TILES);
  copyData(&splitInfoC, FULL_TO_TILES);

  if (OMZTensorSplitDebug) {
    gettimeofday(&start_t1, NULL);
  }

  // Call zdnn_matmul_op on each tile.
  // Iterate over the tiles along the first dim of A.
  for (uint32_t i = 0; i < splitInfoA.numOfTiles; ++i) {
    zdnn_ztensor *zaTensor = splitInfoA.tiles + i;
    zdnn_ztensor *zyTensor = splitInfoY.tiles + i;

    SplitInfo splitInfoYB = {.fullZTensor = zyTensor,
        .axis = E1,
        .numOfElemsPerTile = OMZTensorSplitSize};
    initSplitInfo(&splitInfoYB, true, "MatMul YB");

    // Iterate over the tiles along the second dim of B.
    for (uint32_t j = 0; j < splitInfoB.numOfTiles; ++j) {
      zdnn_ztensor *zbTensor = splitInfoB.tiles + j;
      zdnn_ztensor *zcTensor = splitInfoC.tiles + j;
      zdnn_ztensor *zybTensor = splitInfoYB.tiles + j;
      zdnn_status status = call_zdnn_matmul_op(
          zaTensor, zbTensor, zcTensor, opType, zybTensor, isBcast);
      assert(status == ZDNN_OK);
      if (OMZTensorSplitDebug) {
        int cpuId = 0;
#ifdef __MVS__
        _Cpuid cpuIdWorkArea;
        cpuId = __get_cpuid(cpuIdWorkArea);
#else
        cpuId = sched_getcpu();
#endif
        printf("thread [%u, %u] is on cpu %d\n", i, j, cpuId);
      }
    }
    copyData(&splitInfoYB, TILES_TO_FULL);
    FreeSplitInfoData(&splitInfoYB);
  }

  if (OMZTensorSplitDebug) {
    gettimeofday(&end_t1, NULL);
    totalTime = GetElapseTime(start_t1, end_t1);
    printf("[MatMul] mm loop time, %f (milliseconds)\n", totalTime);
  }

  // Copy data from the tiles back to the full ztensor.
  copyData(&splitInfoY, TILES_TO_FULL);

  // Free temporary buffers.
  FreeSplitInfoData(&splitInfoA);
  FreeSplitInfoData(&splitInfoB);
  FreeSplitInfoData(&splitInfoC);
  FreeSplitInfoData(&splitInfoY);

  if (OMZTensorSplitDebug) {
    gettimeofday(&end_t, NULL);
    totalTime = GetElapseTime(start_t, end_t);
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
  assert(status == ZDNN_OK);
  return status;
}

#ifdef __cplusplus
}
#endif
