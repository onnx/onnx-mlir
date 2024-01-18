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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG 1
#define USE_PTHREAD 1

struct mmStruct {
  zdnn_ztensor *input_a;
  zdnn_ztensor *input_b;
  zdnn_ztensor *input_c;
  zdnn_matmul_ops op_type;
  zdnn_ztensor *output;
};

void *call_zdnn_matmul_op(void *args) {
  struct mmStruct *p = (struct mmStruct *)args;
  zdnn_status status =
      zdnn_matmul_op(p->input_a, p->input_b, p->input_c, p->op_type, p->output);
  return (void *)status;
}

zdnn_status zdnn_matmul_op_ext(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, zdnn_ztensor *output) {
  if (DEBUG) {
    printf("I am in zdnn_matmul_op_ext\n");
    const zdnn_tensor_desc *desc = input_a->transformed_desc;
    printf("A: [%d, %d, %d, %d], %d, ", desc->dim4, desc->dim3, desc->dim2,
        desc->dim1, desc->layout);
    desc = input_b->transformed_desc;
    printf("B: [%d, %d, %d, %d], %d, ", desc->dim4, desc->dim3, desc->dim2,
        desc->dim1, desc->layout);
    desc = input_c->transformed_desc;
    printf("C: [%d, %d, %d, %d], %d.", desc->dim4, desc->dim3, desc->dim2,
        desc->dim1, desc->layout);
    desc = output->transformed_desc;
    printf("Output: [%d, %d, %d, %d], %d\n", desc->dim4, desc->dim3, desc->dim2,
        desc->dim1, desc->layout);
  }

  struct mmStruct *args = malloc(sizeof(struct mmStruct));
  args->input_a = input_a;
  args->input_b = input_b;
  args->input_c = input_c;
  args->op_type = op_type;
  args->output = output;

  zdnn_status status;
  if (USE_PTHREAD) {
    printf("Using pthread\n");
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, call_zdnn_matmul_op, (void *)args);
    pthread_join(thread_id, &status);
  } else {
    printf("No pthread\n");
    status = (zdnn_status)call_zdnn_matmul_op(args);
  }

  free(args);
  assert(status == ZDNN_OK && "Failed to call zdnn_matmul_op");

  return status;
}

#ifdef __cplusplus
}
#endif
