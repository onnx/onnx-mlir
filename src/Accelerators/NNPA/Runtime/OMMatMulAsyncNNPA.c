/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMMatmulNNPA.c ---------------------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// Onnx Matmul op on NNPA Accelerator Runtime
//
//===----------------------------------------------------------------------===//

// Include pthreads (need special treatment on Zos).
#ifdef __MVS__
#define _OPEN_THREADS
#endif
#define _GNU_SOURCE

#include <assert.h>
#include <err.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "OMMatMulAsyncNNPA.h"
#include "onnx-mlir/Runtime/OMTensor.h"
#include "onnx-mlir/Runtime/OnnxDataType.h"
#include "zdnn.h"

#undef USE_THREAD
#define SET_THREAD_AFFINITY
#undef PRINT_TIME

#ifdef __cplusplus
extern "C" {
#endif

struct emit_zdnn_matmul_op_args {
  void *a;
  void *b;
  void *c;
  void *y;
  int64_t broadcast;
};

#ifdef SET_THREAD_AFFINITY
// todo: parse /proc/cpuinfo at runtime
int thread_affinity[] = {0, 1, 2, 3, 4, 5, 6, 7};
// static const int thread_affinity[] = {0, 1, 2, 3, 0, 1, 2, 3};
int zaiu_cpuid_from[] = {0, 7, 14, 21, 27, 33, 39, 45};
int zaiu_cpuid_to[] = {6, 13, 20, 26, 32, 38, 44, 50};
// int zaiu_cpuid_from[] = {0, 12};
// int zaiu_cpuid_to[] = {11, 23};
#endif

//
// Emit zdnn_matmul_op or zdnn_matmul_bcast_op according to the dim_s value.
// If dim_s == 0: Emit zdnn_matmul_op(unstacked)
// If dim_s > 0: Emit zdnn_matmul_op(stacked)
// If dim_s < 0: Emit zdnn_matmul_bcast_op
//
void *emit_zdnn_matmul_op(void *_args) {
  struct emit_zdnn_matmul_op_args *args =
      (struct emit_zdnn_matmul_op_args *)_args;
#ifdef PRINT_TIME
  struct timeval stime, etime, ttime;
  gettimeofday(&stime, NULL);
#endif
  zdnn_status status;
  // perform matrix multiplication between the two input tensors
  if (args->broadcast) {
    printf("AAA0\n");
    fflush(stdout);
    status = zdnn_matmul_bcast_op(
        args->a, args->b, args->c, MATMUL_BCAST_OP_ADDITION, args->y);
    printf("AAA1\n");
    fflush(stdout);
  } else {
    printf("AAA2\n");
    fflush(stdout);
    status =
        zdnn_matmul_op(args->a, args->b, args->c, MATMUL_OP_ADDITION, args->y);
    printf("AAA3\n");
    fflush(stdout);
  }
  assert(status == ZDNN_OK);
#ifdef PRINT_TIME
  gettimeofday(&etime, NULL);
  timersub(&etime, &stime, &ttime);
  printf("emit_zdnn_matmul_op Time: %ld.%06ld sec\n", (long int)ttime.tv_sec,
      (long int)ttime.tv_usec);
  fflush(stdout);
#endif
  printf("AAA4\n");
  fflush(stdout);
  return NULL;
}

//
// Calculate matrix multiplication asynchronously: Y = A * B
// omTensorAsyncWait need to be called before asscsing the results.
//
// static int threadCount = 0;
void omTensorMatMulAsync(void *Y, OMThreadHandler *threadHdr, void *A, void *B,
    void *C, int64_t broadcast) {
  printf("XXXX omTensorMatMulAsync(%p, %p, %p, %p, %p): called\n", Y, threadHdr,
      A, B, C);
  fflush(stdout);
  // OMThreadHandler *threadHdr = omTensorGetDataPtr(threadTensor);
  struct emit_zdnn_matmul_op_args *args =
      (struct emit_zdnn_matmul_op_args *)(&threadHdr->threadArgs);
  printf("XXX0 args=%p\n", args);
  fflush(stdout);
  args->a = A;
  args->b = B;
  args->c = C;
  args->y = Y;
  args->broadcast = broadcast;
#ifndef USE_THREAD
  printf("XXX1\n");
  fflush(stdout);
  emit_zdnn_matmul_op((void *)args);
  printf("XXX2\n");
  fflush(stdout);
#else
  pthread_create(&threadHdr->threadID, NULL, emit_zdnn_matmul_op, (void *)args);
#ifdef SET_THREAD_AFFINITY
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  const int zaiu_id = thread_affinity[threadHdr->threadID & 7];
  for (int i = zaiu_cpuid_from[zaiu_id]; i <= zaiu_cpuid_to[zaiu_id]; i++)
    CPU_SET(i, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
#endif
  printf("XXXX omTensorMatMulAsync(%p, %p, %p, %p, %p): return\n", Y, threadHdr,
      A, B, C);
  fflush(stdout);
}

void omTensorAsyncWait(
    OMTensor *threadTensor, OMTensor *A, OMTensor *B, OMTensor *C) {
  printf("XXXX omTensorMatMulWait(%p, %p, %p, %p): called\n", threadTensor, A,
      B, C);
  fflush(stdout);
#ifdef USE_THREAD
  OMThreadHandler *threadHdr = omTensorGetDataPtr(threadTensor);
  pthread_join(threadHdr->threadID, NULL);
#endif
  printf("XXXX omTensorMatMulWait(%p, %p, %p, %p): return\n", threadTensor, A,
      B, C);
  fflush(stdout);
}

#ifdef __cplusplus
}
#endif
