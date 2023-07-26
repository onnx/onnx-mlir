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

#define USE_NNPA
#define USE_THREAD
// #undef USE_THREAD
#define SET_THREAD_AFFINITY
#define PRINT_TIME

#ifndef USE_NNPA
#include <unistd.h>
#else
#include "zdnn.h"

//
// private functions defined in libzdnn.a
//

zdnn_status set_zdnn_status(zdnn_status status, const char *func_name,
    const char *file_name, int line_no, const char *format, ...);
#define ZDNN_STATUS(status, format, ...)                                       \
  set_zdnn_status(status, __func__, __FILE__, __LINE__, format, __VA_ARGS__)
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct emit_zdnn_matmul_op_args {
  int dim_s;
  int dim_m;
  int dim_n;
  int dim_p;
  void *data_a;
  void *data_b;
  void *data_c;
  void *data_y;
};

static int threadCount = 0;
//
// Emit zdnn_matmul_op or zdnn_matmul_bcast_op according to the dim_s value.
// If dim_s == 0: Emit zdnn_matmul_op(unstacked)
// If dim_s > 0: Emit zdnn_matmul_op(stacked)
// If dim_s < 0: Emit zdnn_matmul_bcast_op
//
void *emit_zdnn_matmul_op(void *_args) {
  struct emit_zdnn_matmul_op_args *args =
      (struct emit_zdnn_matmul_op_args *)_args;
  int dim_s = args->dim_s;
  int dim_m = args->dim_m;
  int dim_n = args->dim_n;
  int dim_p = args->dim_p;
#ifndef USE_NNPA
#ifdef PRINT_TIME
  struct timeval startTime, endTime, totalTime;
  gettimeofday(&startTime, NULL);
#endif
  float *a = (float *)args->data_a;
  float *b = (float *)args->data_b;
  float *c = (float *)args->data_c;
  float *y = (float *)args->data_y;

  if (dim_s == 0) { // unstacked case
    // a[m, n] * b[n, p] + c[p] = y[m, p]
    for (int m = 0; m < dim_m; m++) {
      for (int p = 0; p < dim_p; p++) {
        float ans = 0.0;
        for (int n = 0; n < dim_n; n++) {
          ans += a[m * dim_n + n] * b[n * dim_p + p];
        }
        ans += c[p];
        y[m * dim_p + p] = ans;
      }
    }
  } else if (dims_s > 0) { // stacked case
    // a[s, m, n] * b[s, n, p] + c[s, p] = y[s, m, p]
    for (int s = 0; s < dim_s; s++) {
      for (int m = 0; m < dim_m; m++) {
        for (int p = 0; p < dim_p; p++) {
          float ans = 0.0;
          for (int n = 0; n < dim_n; n++) {
            ans += a[s * (dim_m * dim_n) + m * dim_n + n] *
                   b[s * (dim_n * dim_p) + n * dim_p + p];
          }
          ans += c[s * dim_p + p];
          y[s * (dim_m * dim_p) + m * dim_p + p] = ans;
        }
      }
    }
  } else { // bcast case
    dim_s = -dim_s;
    // a[s, m, n] * b[n, p] + c[p] = y[s, m, p]
    for (int s = 0; s < dim_s; s++) {
      for (int m = 0; m < dim_m; m++) {
        for (int p = 0; p < dim_p; p++) {
          float ans = 0.0;
          for (int n = 0; n < dim_n; n++) {
            ans += a[s * (dim_m * dim_n) + m * dim_n + n] * b[n * dim_p + p];
          }
          ans += c[p];
          y[s * (dim_m * dim_p) + m * dim_p + p] = ans;
        }
      }
    }
  }
#ifdef PRINT_TIME
  gettimeofday(&endTime, NULL);
  timersub(&endTime, &startTime, &totalTime);
  printf("MatMul C code Time elapsed: %ld.%06ld sec = %ld.06ld (stick) + "
         "%ld.%06ld (op) "
         " %ld.06ld (unstick)\n",
      (long int)totalTime.tv_sec, (long int)totalTime.tv_usec);
#endif
#else // if defined(USE_NNPA)
#ifdef PRINT_TIME
  struct timeval startTime, stickEndTime, opEndTime, unstickEndTime;
  struct timeval stickTime, opTime, unstickTime, totalTime;
  gettimeofday(&startTime, NULL);
#endif
  bool is_bcast = false;
  if (args->dim_s < 0) { // zdnn_matmul_bcast_op case
    dim_s = -args->dim_s;
    is_bcast = true;
  }
  void *data_a = args->data_a;
  void *data_b = args->data_b;
  void *data_c = args->data_c;
  void *data_y = args->data_y;
  zdnn_tensor_desc pre_tfrmd_desc_a, pre_tfrmd_desc_b, pre_tfrmd_desc_c,
      pre_tfrmd_desc_y;
  zdnn_tensor_desc tfrmd_desc_a, tfrmd_desc_b, tfrmd_desc_c, tfrmd_desc_y;
  zdnn_ztensor ztensor_a, ztensor_b, ztensor_c, ztensor_y;
  zdnn_data_types type = FP32;
  zdnn_matmul_ops ops = NNPA_MATMUL_OP_ADDITION;
  zdnn_status status;
  int64_t size;
  // generate transformed shape information for input 1, input 2 and output
  zdnn_init_pre_transformed_desc(
      ZDNN_2D, type, &pre_tfrmd_desc_a, dim_m, dim_n);
  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc_a, &tfrmd_desc_a);
  assert(status == ZDNN_OK);
  zdnn_init_pre_transformed_desc(
      ZDNN_2D, type, &pre_tfrmd_desc_b, dim_n, dim_p);
  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc_b, &tfrmd_desc_b);
  assert(status == ZDNN_OK);
  zdnn_init_pre_transformed_desc(ZDNN_1D, type, &pre_tfrmd_desc_c, dim_p);
  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc_c, &tfrmd_desc_c);
  assert(status == ZDNN_OK);
  zdnn_init_pre_transformed_desc(
      ZDNN_2D, type, &pre_tfrmd_desc_y, dim_m, dim_p);
  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc_y, &tfrmd_desc_y);
  assert(status == ZDNN_OK);

  // initialize zTensors and allocate 4k-aligned storage via helper function
  zdnn_init_ztensor(&pre_tfrmd_desc_a, &tfrmd_desc_a, &ztensor_a);
  zdnn_init_ztensor(&pre_tfrmd_desc_b, &tfrmd_desc_b, &ztensor_b);
  zdnn_init_ztensor(&pre_tfrmd_desc_c, &tfrmd_desc_c, &ztensor_c);
  zdnn_init_ztensor(&pre_tfrmd_desc_y, &tfrmd_desc_y, &ztensor_y);

  // Allocate zTensor buffers
  size = zdnn_getsize_ztensor(ztensor_a.transformed_desc);
  ztensor_a.buffer_size = size;
  if (posix_memalign(&ztensor_a.buffer, 4096, size))
    assert("Unable to allocate buffer for ztensor_a");

  size = zdnn_getsize_ztensor(ztensor_b.transformed_desc);
  ztensor_b.buffer_size = size;
  if (posix_memalign(&ztensor_b.buffer, 4096, size))
    assert("Unable to allocate buffer for ztensor_b");

  size = zdnn_getsize_ztensor(ztensor_c.transformed_desc);
  ztensor_c.buffer_size = size;
  if (posix_memalign(&ztensor_c.buffer, 4096, size))
    assert("Unable to allocate buffer for ztensor_c");

  size = zdnn_getsize_ztensor(ztensor_y.transformed_desc);
  ztensor_y.buffer_size = size;
  if (posix_memalign(&ztensor_y.buffer, 4096, size))
    assert("Unable to allocate buffer for ztensor_y");

  // transform the feature tensor
  status = zdnn_transform_ztensor(&ztensor_a, data_a);
  assert(status == ZDNN_OK);
  status = zdnn_transform_ztensor(&ztensor_b, data_b);
  assert(status == ZDNN_OK);
  status = zdnn_transform_ztensor(&ztensor_c, data_c);
  assert(status == ZDNN_OK);
#ifdef PRINT_TIME
  gettimeofday(&stickEndTime, NULL);
#endif

  // perform matrix multiplication between the two input tensors
  if (is_bcast)
    status =
        zdnn_matmul_op(&ztensor_a, &ztensor_b, &ztensor_c, ops, &ztensor_y);
  else
    status = zdnn_matmul_bcast_op(
        &ztensor_a, &ztensor_b, &ztensor_c, ops, &ztensor_y);
  assert(status == ZDNN_OK);
#ifdef PRINT_TIME
  gettimeofday(&opEndTime, NULL);
#endif

  // transform resultant zTensor back to original data format
  status = zdnn_transform_origtensor(&ztensor_y, data_y);
  assert(status == ZDNN_OK);

  // Free zTensor buffers
  free(ztensor_a.buffer);
  free(ztensor_b.buffer);
  free(ztensor_c.buffer);
  free(ztensor_y.buffer);
#ifdef PRINT_TIME
  gettimeofday(&unstickEndTime, NULL);
  timersub(&unstickEndTime, &startTime, &totalTime);
  timersub(&stickEndTime, &startTime, &stickTime);
  timersub(&opEndTime, &unstickEndTime, &opTime);
  timersub(&unstickEndTime, &opEndTime, &unstickTime);
  printf("MatMul C code Time elapsed: %ld.%06ld sec = %ld.06ld (stick) + "
         "%ld.%06ld (op) "
         " %ld.%06ld (unstick)\n",
      (long int)totalTime.tv_sec, (long int)totalTime.tv_usec,
      (long int)stickTime.tv_sec, (long int)stickTime.tv_usec,
      (long int)opTime.tv_sec, (long int)opTime.tv_usec,
      (long int)unstickTime.tv_sec, (long int)unstickTime.tv_usec);
#endif
#endif
  return NULL;
}

#ifdef USE_NNPA
static int zdnn_init_done = 0;
#endif

//
// Calculate matrix multiplication asynchronously: Y = A * B
// omTensorAsyncWait need to be called before asscsing the results.
//
// static int threadCount = 0;
void omTensorMatMulAsync(OMTensor *Y, OMTensor *threadTensor, OMTensor *A,
    OMTensor *B, OMTensor *C) {
  OMThreadHandler *threadHdr = omTensorGetDataPtr(threadTensor);
  const OM_DATA_TYPE dataType = omTensorGetDataType(A);
  assert(dataType == ONNX_TYPE_FLOAT &&
         "omTensorMatmul assumes ONNX_TYPE_FLOAT type");
  const int64_t rankA = omTensorGetRank(A);
  const int64_t rankB = omTensorGetRank(B);
  const int64_t rankC = omTensorGetRank(C);
  const int64_t rankY = omTensorGetRank(Y);
  assert((((rankA == 2) && (rankB == 2) && (rankC == 1) &&
              (rankY == 2)) || // unstacked
             ((rankA == 3) && (rankB == 3) && (rankC == 2) &&
                 (rankY == 3)) || // stacked
             ((rankA == 3) && (rankB == 2) && (rankC == 1) &&
                 (rankY == 3))) && // bcast
         "omTensorMatmul: inconsistent ranks of input/output tensors");
  const int64_t *shapeA = omTensorGetShape(A);
  const int64_t *shapeB = omTensorGetShape(B);
  const int64_t *shapeC = omTensorGetShape(C);
  const int64_t *shapeY = omTensorGetShape(Y);
  void *dataA = omTensorGetDataPtr(A);
  void *dataB = omTensorGetDataPtr(B);
  void *dataC = omTensorGetDataPtr(C);
  void *dataY = omTensorGetDataPtr(Y);
  assert(shapeA[0] == shapeY[0] &&
         "omTensorMatmul: inconsistent input shapes (dim_m)");
  assert(shapeA[1] == shapeB[0] &&
         "omTensorMatmul: inconsistent input shapes (dim_n)");
  assert(shapeB[1] == shapeY[1] && shapeB[1] == shapeC[0] &&
         "omTensorMatmul: inconsistent input shapes (dim_p)");
  int dim_s, dim_m, dim_n, dim_p;
  if (rankA == 2) { // zdnn_matmul_op(unstacked) case
    dim_s = 0;
    dim_m = shapeA[0];
    dim_n = shapeA[1];
    dim_p = shapeB[1];
  }
  if (rankB == 3) { // zdmm_matmul_op(stacked) case
    dim_s = shapeA[0];
    dim_m = shapeA[1];
    dim_n = shapeA[2];
    dim_p = shapeB[2];
  } else { // zdnn_matmul_bcast_op case
    dim_s = shapeA[0];
    dim_m = shapeA[1];
    dim_n = shapeA[2];
    dim_p = shapeB[1];
  }
  // transfer inputs into ztensor, call zdnn_matmul_op, and transfer outputs
  // from ztensor to normal buffer
  struct emit_zdnn_matmul_op_args *args =
      (struct emit_zdnn_matmul_op_args *)(&threadHdr->threadArgs);
  args->dim_s = dim_s;
  args->dim_m = dim_m;
  args->dim_n = dim_n;
  args->dim_p = dim_p;
  args->data_a = dataA;
  args->data_b = dataB;
  args->data_c = dataC;
  args->data_y = dataY;
#ifndef USE_THREAD
  emit_zdnn_matmul_op((void *)args);
#else
  pthread_create(&threadHdr->threadID, NULL, emit_zdnn_matmul_op, (void *)args);
#ifdef SET_THREAD_AFFINITY
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  int cpu_base = (8 * threadCount++) % 16;
  for (int i = cpu_base; i < (cpu_base + 8); i++)
    CPU_SET(i, &cpuset);
  pthread_setaffinity_np(&threadHdr->threadID, sizeof(cpuset), &cpuset);
/*
int s;
s = pthread_getaffinity_np(&threadHdr->threadID, sizeof(cpuset), &cpuset);
printf("Set returned by pthread_getaffinity_np() contained:\n");
for (size_t j = 0; j < CPU_SETSIZE; j++)
  if (CPU_ISSET(j, &cpuset))
    printf("    CPU %zu\n", j);
*/
#endif
#endif
}

void omTensorAsyncWait(
    OMTensor *threadTensor, OMTensor *A, OMTensor *B, OMTensor *C) {
#ifdef USE_THREAD
  OMThreadHandler *threadHdr = omTensorGetDataPtr(threadTensor);
  pthread_join(threadHdr->threadID, NULL);
#endif
}

#ifdef __cplusplus
}
#endif
