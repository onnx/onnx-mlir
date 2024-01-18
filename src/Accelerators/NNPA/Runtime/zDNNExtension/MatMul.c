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

// Include pthreads (need special treatment on Zos).
#ifdef __MVS__
#define _OPEN_THREADS
#endif
#include <pthread.h>

#include <stdio.h>

#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG 1

zdnn_status zdnn_matmul_op_ext(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, zdnn_ztensor *output) {
  if (DEBUG) {
    printf("I am in zdnn_matmul_op_ext\n");
    // Examine input_a.
    const zdnn_tensor_desc *desc = input_a->transformed_desc;
    printf("A: [%d, %d, %d, %d], %d\n", desc->dim4, desc->dim3, desc->dim2,
        desc->dim1, desc->layout);
    desc = input_b->transformed_desc;
    printf("B: [%d, %d, %d, %d], %d\n", desc->dim4, desc->dim3, desc->dim2,
        desc->dim1, desc->layout);
    desc = input_c->transformed_desc;
    printf("C: [%d, %d, %d, %d], %d\n", desc->dim4, desc->dim3, desc->dim2,
        desc->dim1, desc->layout);
    desc = output->transformed_desc;
    printf("Output: [%d, %d, %d, %d], %d\n", desc->dim4, desc->dim3, desc->dim2,
        desc->dim1, desc->layout);
  }
  zdnn_status status =
      zdnn_matmul_op(input_a, input_b, input_c, op_type, output);
  return status;
}

#ifdef __cplusplus
}
#endif
