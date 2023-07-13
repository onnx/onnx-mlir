/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OMAsyncMatMul.cpp - OMTensor C/C++ Implementation ----------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ neutral implementation of OMTensorList data
// structures and helper functions.
//
//===----------------------------------------------------------------------===//
#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>
// #include <vector>

#include "onnx-mlir/Runtime/OMTensor.h"

void omTensorMatMulAsync(
    OMTensor *Y, OMTensor *tick, OMTensor *A, OMTensor *B) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(A);
  assert(dataType == ONNX_TYPE_FLOAT &&
         "omTensorMatmul assumes ONNX_TYPE_FLOAT type");
  assert((omTensorGetRank(A) == 2) && (omTensorGetRank(A) == 2) &&
         (omTensorGetRank(A) == 2) && "omTensorMatmul assumes rank 2 tensors");
  const int64_t *shapeA = omTensorGetShape(A);
  const int64_t *shapeB = omTensorGetShape(B);
  const int64_t *shapeY = omTensorGetShape(Y);
  void *dataA = omTensorGetDataPtr(A);
  void *dataB = omTensorGetDataPtr(B);
  void *dataY = omTensorGetDataPtr(Y);
  assert(shapeA[1] == shapeB[0] && "omTensorMatmul: inconsistent input shapes");
  int dim_m = shapeA[0];
  int dim_n = shapeA[1];
  int dim_p = shapeB[1];
  printf("dim_m = %d, dim_n = %d, dim_p = %d\n", dim_m, dim_n, dim_p);
  // Compute Matmul A(mxn) * B(nxp).
  for (int64_t m = 0; m < dim_m; ++m) {
    for (int64_t p = 0; p < dim_p; ++p) {
      float *y = (float *)dataY + m * dim_p + p;
      *y = 0;
      for (int64_t n = 0; n < dim_n; n++) {
        float *tmpA = (float *)dataA + m * dim_n + n;
        float *tmpB = (float *)dataB + n * dim_p + p;
        *y += (*tmpA) * (*tmpB);
      }
    }
  }
}

#if 0
//
// Calculate matrix multiplication asynchronously: Y = A * B
// omTensorAsyncWait need to be called before asscsing the results.
//
void omTensorMatMulAsync(OMTensor *Y, uint64_t *threadID, OMTensor *A, OMTensor *B,
                         OMTensor *C) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(A);
  assert(dataType == ONNX_TYPE_FLOAT && "omTensorMatmul assumes ONNX_TYPE_FLOAT type");
  assert((omTensorGetRank(A) == 2) && (omTensorGetRank(A) == 2) &&
      (omTensorGetRank(A) == 2) && "omTensorMatmul assumes rank 2 tensors");
  const int64_t *shapeA = omTensorGetShape(A);
  const int64_t *shapeB = omTensorGetShape(B);
  const int64_t *shapeY = omTensorGetShape(Y);
  void *dataA = omTensorGetDataPtr(A);
  void *dataB = omTensorGetDataPtr(B);
  void *dataY = omTensorGetDataPtr(Y);
  assert(shapeA[1] == shapeB[0] && "omTensorMatmul: inconsistent input shapes");
  int dim_m = shapeA[0];
  int dim_n = shapeA[1];
  int dim_p = shapeB[1];
  zdnn_tensor_desc pre_tfrmd_desc_1, pre_tfrmd_desc_2, pre_tfrmd_desc_3,
      pre_tfrmd_desc_out;
  zdnn_tensor_desc tfrmd_desc_1, tfrmd_desc_2, tfrmd_desc_3, tfrmd_desc_out;
  zdnn_ztensor ztensor_a, ztensor_b, ztensor_c, ztensor_out;
  zdnn_data_types type = FP32;
  zdnn_status status;
  // generate transformed shape information for input 1, input 2 and output
  zdnn_init_pre_transformed_desc(ZDNN_2D, type, &pre_tfrmd_desc_1, dim_m, dim_n);
  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc_1, &tfrmd_desc_1);
  assert(status == ZDNN_OK);
  zdnn_init_pre_transformed_desc(ZDNN_2D, type, &pre_tfrmd_desc_2, dim_n, dim_p);
  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc_2, &tfrmd_desc_2);
  assert(status == ZDNN_OK);
  zdnn_init_pre_transformed_desc(ZDNN_1D, type, &pre_tfrmd_desc_3, dim_p);
  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc_3, &tfrmd_desc_3);
  assert(status == ZDNN_OK);
  zdnn_init_pre_transformed_desc(ZDNN_2D, type, &pre_tfrmd_desc_out, dim_m, dim_p);
  status = zdnn_generate_transformed_desc(
                                          &pre_tfrmd_desc_out, &tfrmd_desc_out);
  assert(status == ZDNN_OK);

  // initialize zTensors and allocate 4k-aligned storage via helper function
  status = zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc_1, &tfrmd_desc_1, &ztensor_a);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc_2, &tfrmd_desc_2, &ztensor_b);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc_3, &tfrmd_desc_3, &ztensor_c);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc_out, &tfrmd_desc_out, &ztensor_out);
  assert(status == ZDNN_OK);

  // transfer inputs into ztensor, call zdnn_matmul_op, and transfer outputs
  // from ztensor to normal buffer
  struct emit_zdnn_matmul_op_args args = {
      ztensor_a, ztensor_b, ztensor_c, dataA, dataB, dataY,
      NNPA_MATMUL_OP_ADDITION, ztensor_out, dataY, status};
#ifdef CALL_EMIT_ZDNN_MATMUL_OP
  emit_zdnn_matmul_op((void *) &args);
#else
  pthread_create((pthread_t *) threadID, NULL, emit_zdnn_matmul_op,
        (void *) &args);
#endif
}
#endif
