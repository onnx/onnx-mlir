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
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "OMMatMulAsyncNNPA.h"
#include "onnx-mlir/Runtime/OMTensor.h"
#include "onnx-mlir/Runtime/OnnxDataType.h"

// #define USE_NNPA
#define USE_THREAD
// #undef USE_THREAD

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

#define MAX_THREAD_NUM 32

#ifdef __cplusplus
extern "C" {
#endif

struct emit_zdnn_matmul_op_args {
  int dim_m;
  int dim_n;
  int dim_p;
  void *data_a;
  void *data_b;
  void *data_c;
  void *data_y;
};

void *emit_zdnn_matmul_op(void *_args) {
  struct emit_zdnn_matmul_op_args *args =
      (struct emit_zdnn_matmul_op_args *)_args;
  int dim_m = args->dim_m;
  int dim_n = args->dim_n;
  int dim_p = args->dim_p;
  printf("emit_zdnn  m %d, n %d, p %d\n", dim_m, dim_n, dim_p);
#ifndef USE_NNPA
  // wait random(0.0-1.0) sec to simulate NNPA execution
  // usleep(random() % 1000000);
  float *a = (float *)args->data_a;
  float *b = (float *)args->data_b;
  float *c = (float *)args->data_c;
  float *y = (float *)args->data_y;
  for (int i = 0; i < dim_m; i++) {
    for (int k = 0; k < dim_n; k++) {
      if (a[i * dim_n + k] != 1.0) {
        printf("i,k= %d, %d; a[i * dim_n + k] = %f\n", i, k, a[i * dim_n + k]);
        // a[i * dim_n + k] = 1.0;
      }
    }
  }
  for (int j = 0; j < dim_p; j++) {
    for (int k = 0; k < dim_n; k++) {
      if (b[k * dim_p + j] != 1.0) {
        printf("k, j= %d, %d; b[k * dim_p + j] = %f\n", k, j, b[k * dim_p + j]);
        // b[k * dim_p + j] = 1.0;
      }
    }
  }
  for (int p = 0; p < dim_p; p++) {
    if (c[p] != 0.0) {
      printf("p= %d; c[p] = %f\n", p, c[p]);
      // c[p] = 0.0;
    }
  }

  for (int i = 0; i < dim_m; i++) {
    for (int j = 0; j < dim_p; j++) {
      float ans = 0.0;
      for (int k = 0; k < dim_n; k++) {
        ans += a[i * dim_n + k] * b[k * dim_p + j];
      }
      ans += c[j];
      if (ans != 2.0)
        printf("i,j,ans = %d, %d, %f\n", i, j, ans);
      y[i * dim_p + j] = ans;
    }
  }
#else
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

  // perform matrix multiplication between the two input tensors
  status = zdnn_matmul_op(&ztensor_a, &ztensor_b, &ztensor_c, ops, &ztensor_y);
  assert(status == ZDNN_OK);

  // transform resultant zTensor back to original data format
  status = zdnn_transform_origtensor(&ztensor_y, data_y);
  assert(status == ZDNN_OK);

  // Free zTensor buffers
  free(ztensor_a.buffer);
  free(ztensor_b.buffer);
  free(ztensor_c.buffer);
  free(ztensor_y.buffer);
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
void omTensorMatMulAsync(OMTensor *Y, OMTensor *threadTensor, OMTensor *A,
    OMTensor *B, OMTensor *C) {
  printf("Call C code\n");
  OMThreadHandler *threadHdr = omTensorGetDataPtr(threadTensor);
  const OM_DATA_TYPE dataType = omTensorGetDataType(A);
  assert(dataType == ONNX_TYPE_FLOAT &&
         "omTensorMatmul assumes ONNX_TYPE_FLOAT type");
  assert((omTensorGetRank(A) == 2) && (omTensorGetRank(A) == 2) &&
         (omTensorGetRank(A) == 2) && "omTensorMatmul assumes rank 2 tensors");
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
  int dim_m = shapeA[0];
  int dim_n = shapeA[1];
  int dim_p = shapeB[1];
#ifdef USE_NNPA
  if (zdnn_init_done == 0) {
    zdnn_init();
    zdnn_init_done++;
  }
#endif
  printf("Call C code. m %d, n %d, p %d\n", dim_m, dim_n, dim_p);
  // transfer inputs into ztensor, call zdnn_matmul_op, and transfer outputs
  // from ztensor to normal buffer
  struct emit_zdnn_matmul_op_args *args =
      (struct emit_zdnn_matmul_op_args *)(&threadHdr->threadArgs);
  args->dim_m = dim_m;
  args->dim_n = dim_n;
  args->dim_p = dim_p;
  args->data_a = dataA;
  args->data_b = dataB;
  args->data_c = dataC;
  args->data_y = dataY;

#ifndef USE_THREAD
  printf("Call C code. NO thread\n");
  emit_zdnn_matmul_op((void *)args);
#else
  printf("Call ThreadHdr %p\n", threadHdr);
  pthread_create(&threadHdr->threadID, NULL, emit_zdnn_matmul_op, (void *)args);
#endif
}

void omTensorAsyncWait(
    OMTensor *threadTensor, OMTensor *A, OMTensor *B, OMTensor *C) {
#ifdef USE_THREAD
  OMThreadHandler *threadHdr = omTensorGetDataPtr(threadTensor);
  printf("Wait ThreadHdr %p\n", threadHdr);
  pthread_join(threadHdr->threadID, NULL);
#endif
}

#ifdef __cplusplus
}
#endif
