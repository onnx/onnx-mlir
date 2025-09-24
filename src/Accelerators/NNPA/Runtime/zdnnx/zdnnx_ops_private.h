/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ zdnnx_ops_private.h -------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Sets of private data structures and functions for defining operations in the
// zdnn extension library.
//
// These private information are shared among seq_ops.c and omp_ops.c.
//
//===----------------------------------------------------------------------===//

#ifndef ZDNNX_ZDNNX_OPS_PRIVATE_H
#define ZDNNX_ZDNNX_OPS_PRIVATE_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "zdnnx.h"

typedef enum ElemementwiseOp {
  // Binary
  ZDNNX_ADD_OP = 0,
  ZDNNX_DIV_OP = 1,
  ZDNNX_MAX_OP = 2,
  ZDNNX_MIN_OP = 3,
  ZDNNX_MUL_OP = 4,
  ZDNNX_SUB_OP = 5,
  // Unary
  ZDNNX_EXP_OP = 50,
  ZDNNX_GELU_OP = 51,
  ZDNNX_INVSQRT_OP = 52,
  ZDNNX_LEAKY_RELU_OP = 53,
  ZDNNX_LOG_OP = 54,
  ZDNNX_RELU_OP = 55,
  ZDNNX_SIGMOID_OP = 56,
  ZDNNX_SQRT_OP = 57,
  ZDNNX_TANH_OP = 58,
} ElemementwiseOp;

#ifdef ZDNNX_DEBUG
#define ZDNNX_START_TIMING()                                                   \
  double total_time = 0.;                                                      \
  struct timeval start_t = {0}, end_t = {0};                                   \
  gettimeofday(&start_t, NULL);
#else
#define ZDNNX_START_TIMING()
#endif

#ifdef ZDNNX_DEBUG
#define ZDNNX_STOP_TIMING(msg)                                                 \
  gettimeofday(&end_t, NULL);                                                  \
  total_time = (((end_t.tv_sec * 1000000.) + end_t.tv_usec) -                  \
                   ((start_t.tv_sec * 1000000) + start_t.tv_usec)) /           \
               1000;                                                           \
  printf("[%s] total time, %f (milliseconds)\n", msg, total_time);
#else
#define ZDNNX_STOP_TIMING(msg)
#endif

#ifdef ZDNNX_WITH_OMP
#define ZDNNX_CALL_FUNC(msg, seq_func, omp_func, ...)                          \
  ZDNNX_START_TIMING();                                                        \
  if (zdnnx_get_num_zaiu_threads() == 1)                                       \
    status = seq_func(__VA_ARGS__);                                            \
  else                                                                         \
    status = omp_func(__VA_ARGS__);                                            \
  ZDNNX_STOP_TIMING(msg);
#else
#define ZDNNX_CALL_FUNC(msg, seq_func, omp_func, ...)                          \
  ZDNNX_START_TIMING();                                                        \
  status = seq_func(__VA_ARGS__);                                              \
  ZDNNX_STOP_TIMING(msg);
#endif

#define ZDNNX_CHECK_STATUS(status, zdnn_name)                                  \
  zdnnx_check_status((status), (zdnn_name))

// Get the max number of elements per dim on zAIU.
uint32_t zdnnx_get_nnpa_max_dim_size(zdnnx_axis dim_index);

// Get the max number of elements per tensor on zAIU.
uint64_t zdnnx_get_nnpa_max_tensor_size();

/**
 * \brief Check zdnn status
 *
 * Check if the zdnn status is not a zdnn_ok and print out the
 * status message along with the error
 *
 * @param status zdnn status
 * @param zdnn_name name of the zdnn api
 */
void zdnnx_check_status(zdnn_status status, const char *zdnn_name);

#endif
