/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- zdnnx_ops.c -------------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Operations in the zdnn extension to replace zdnn operations.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef ZDNNX_WITH_OMP
#include "omp_ops.h"
#include <omp.h>
#endif

#include "seq_ops.h"
#include "zdnnx.h"
#include "zdnnx_ops.h"

// Keep these values to avoid calling zdnn functions multiple times.
uint32_t nnpa_max_dim_size_e4 = 0;
uint32_t nnpa_max_dim_size_e3 = 0;
uint32_t nnpa_max_dim_size_e2 = 0;
uint32_t nnpa_max_dim_size_e1 = 0;
uint64_t nnpa_max_tensor_size = 0;

#ifdef ZDNNX_DEBUG
#define START_TIMING()                                                         \
  double total_time = 0.;                                                      \
  struct timeval start_t = {0}, end_t = {0};                                   \
  gettimeofday(&start_t, NULL);
#else
#define START_TIMING()
#endif

#ifdef ZDNNX_DEBUG
#define STOP_TIMING(msg)                                                       \
  gettimeofday(&end_t, NULL);                                                  \
  total_time = (((end_t.tv_sec * 1000000.) + end_t.tv_usec) -                  \
                   ((start_t.tv_sec * 1000000) + start_t.tv_usec)) /           \
               1000;                                                           \
  printf("[%s] total time, %f (milliseconds)\n", msg, total_time);
#else
#define STOP_TIMING(msg)
#endif

#ifdef ZDNNX_WITH_OMP
#define CALL_ZDNNX_FUNC(msg, seq_func, omp_func, ...)                          \
  START_TIMING();                                                              \
  if (zdnnx_get_num_procs() == 1)                                              \
    status = seq_func(__VA_ARGS__);                                            \
  else                                                                         \
    status = omp_func(__VA_ARGS__);                                            \
  STOP_TIMING(msg);
#else
#define CALL_ZDNNX_FUNC(msg, seq_func, omp_func, ...)                          \
  START_TIMING();                                                              \
  status = seq_func(__VA_ARGS__);                                              \
  STOP_TIMING(msg);
#endif

uint32_t get_nnpa_max_dim_size(zdnnx_axis dim_index) {
  switch (dim_index) {
  case E4:
    if (nnpa_max_dim_size_e4 == 0)
      nnpa_max_dim_size_e4 = is_telum_1 ? zdnn_get_nnpa_max_dim_idx_size()
                                        : zdnn_get_max_for_dim(4);
    return nnpa_max_dim_size_e4;
  case E3:
    if (nnpa_max_dim_size_e3 == 0)
      nnpa_max_dim_size_e3 = is_telum_1 ? zdnn_get_nnpa_max_dim_idx_size()
                                        : zdnn_get_max_for_dim(3);
    return nnpa_max_dim_size_e3;
  case E2:
    if (nnpa_max_dim_size_e2 == 0)
      nnpa_max_dim_size_e2 = is_telum_1 ? zdnn_get_nnpa_max_dim_idx_size()
                                        : zdnn_get_max_for_dim(2);
    return nnpa_max_dim_size_e2;
  case E1:
    if (nnpa_max_dim_size_e1 == 0)
      nnpa_max_dim_size_e1 = is_telum_1 ? zdnn_get_nnpa_max_dim_idx_size()
                                        : zdnn_get_max_for_dim(1);
    return nnpa_max_dim_size_e1;
  default:
    return 0;
  }
}

uint64_t get_nnpa_max_tensor_size() {
  if (nnpa_max_tensor_size == 0) {
    // zdnn_get_nnpa_max_tensor_size() returns size in bytes.
    nnpa_max_tensor_size = zdnn_get_nnpa_max_tensor_size() / 2;
  }
  return nnpa_max_tensor_size;
}

void check_status(zdnn_status status, const char *zdnn_name) {
  if (OMStatusMessagesEnabled && status != ZDNN_OK) {
    fprintf(stdout, "[zdnnx] %s : %s\n", zdnn_name,
        zdnn_get_status_message(status));
  }
}

static inline zdnn_status call_unary_op(const char *msg,
    const zdnn_ztensor *input, const void *scalar_input, zdnn_ztensor *output,
    ElemementwiseOp op_type) {
  START_TIMING();

  zdnn_status status;
#ifdef ZDNNX_WITH_OMP
  if (zdnnx_get_num_procs() == 1)
    status = seq_unary_elementwise(input, scalar_input, output, op_type);
  else
    status = omp_unary_elementwise(input, scalar_input, output, op_type);
#else
  status = seq_unary_elementwise(input, scalar_input, output, op_type);
#endif

  STOP_TIMING(msg);

  return status;
}

static inline zdnn_status call_binary_op(const char *msg,
    const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
    zdnn_ztensor *output, ElemementwiseOp op_type) {
  START_TIMING();

  zdnn_status status;
#ifdef ZDNNX_WITH_OMP
  if (zdnnx_get_num_procs() == 1)
    status = seq_binary_elementwise(input_a, input_b, output, op_type);
  else
    status = omp_binary_elementwise(input_a, input_b, output, op_type);
#else
  status = seq_binary_elementwise(input_a, input_b, output, op_type);
#endif

  STOP_TIMING(msg);

  return status;
}

// -----------------------------------------------------------------------------
// Extension Functions
// Same name as zdnn functions but with the `zdnnx` prefix instead of `zdnn`.
// -----------------------------------------------------------------------------

zdnn_status zdnnx_matmul_op(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, int op_type,
    zdnn_ztensor *output) {
  zdnn_status status;
  CALL_ZDNNX_FUNC("MatMul", seq_matmul, omp_matmul, input_a, input_b, input_c,
      op_type, output, /*is_bcast=*/false);
  CHECK_ZDNN_STATUS(status, "zdnn_matmul");
  return status;
}

zdnn_status zdnnx_matmul_bcast_op(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, int op_type,
    zdnn_ztensor *output) {
  zdnn_status status;
  CALL_ZDNNX_FUNC("MatMul", seq_matmul, omp_matmul, input_a, input_b, input_c,
      op_type, output, /*is_bcast=*/true);
  CHECK_ZDNN_STATUS(status, "zdnn_matmul_bcast");
  return status;
}

zdnn_status zdnnx_matmul_transpose_op(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, int transpose_a,
    int transpose_b, int opType, zdnn_ztensor *output) {
  zdnn_status status;
  CALL_ZDNNX_FUNC("Transposed MatMul", zdnn_matmul_transpose_op,
      zdnn_matmul_transpose_op, input_a, input_b, input_c, transpose_a,
      transpose_b, opType, output);
  CHECK_ZDNN_STATUS(status, "zdnn_matmul_transpose");
  return status;
}

zdnn_status zdnnx_add(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Add", inputA, inputB, output, ZDNNX_ADD_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_add");
  return status;
}

zdnn_status zdnnx_sub(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Sub", inputA, inputB, output, ZDNNX_SUB_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_sub");
  return status;
}

zdnn_status zdnnx_mul(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Mul", inputA, inputB, output, ZDNNX_MUL_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_mul");
  return status;
}

zdnn_status zdnnx_div(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Div", inputA, inputB, output, ZDNNX_DIV_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_div");
  return status;
}

zdnn_status zdnnx_min(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Min", inputA, inputB, output, ZDNNX_MIN_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_min");
  return status;
}

zdnn_status zdnnx_max(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Max", inputA, inputB, output, ZDNNX_MAX_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_max");
  return status;
}

zdnn_status zdnnx_exp(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status = call_unary_op("Exp", input, NULL, output, ZDNNX_EXP_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_exp");
  return status;
}

zdnn_status zdnnx_log(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status = call_unary_op("Log", input, NULL, output, ZDNNX_LOG_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_log");
  return status;
}

zdnn_status zdnnx_relu(const zdnn_ztensor *input, const void *clipping_value,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Relu", input, clipping_value, output, ZDNNX_RELU_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_relu");
  return status;
}

zdnn_status zdnnx_sigmoid(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Sigmoid", input, NULL, output, ZDNNX_SIGMOID_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_sigmoid");
  return status;
}

zdnn_status zdnnx_tanh(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Tanh", input, NULL, output, ZDNNX_TANH_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_tanh");
  return status;
}

zdnn_status zdnnx_softmax(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output) {
  zdnn_status status;
  CALL_ZDNNX_FUNC(
      "Softmax", seq_softmax, omp_softmax, input, save_area, act_func, output);
  CHECK_ZDNN_STATUS(status, "zdnn_softmax");
  return status;
}

// -----------------------------------------------------------------------------
// Extension Functions for arch15/z17
// arch15/z17 specific zdnn functions but with the `zdnnx` prefix.
// Retrieve the zdnn status message
// -----------------------------------------------------------------------------

zdnn_status zdnnx_gelu(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Gelu", input, NULL, output, ZDNNX_GELU_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_gelu");
  return status;
}

zdnn_status zdnnx_invsqrt(
    const zdnn_ztensor *input, float epsilon, zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Invsqrt", input, &epsilon, output, ZDNNX_INVSQRT_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_invsqrt");
  return status;
}

zdnn_status zdnnx_leaky_relu(const zdnn_ztensor *input,
    const void *clipping_value, float adjustment_factor, zdnn_ztensor *output) {
  zdnn_status status;
  CALL_ZDNNX_FUNC("LeakyRelu", zdnn_leaky_relu, zdnn_leaky_relu, input,
      clipping_value, adjustment_factor, output);
  CHECK_ZDNN_STATUS(status, "zdnn_leakyrelu");
  return status;
}

zdnn_status zdnnx_reduce(const zdnn_ztensor *input, void *save_area, int opType,
    zdnn_ztensor *output) {
  zdnn_status status;
  CALL_ZDNNX_FUNC(
      "Reduce", zdnn_reduce, zdnn_reduce, input, save_area, opType, output);
  CHECK_ZDNN_STATUS(status, "zdnn_reduce");
  return status;
}

zdnn_status zdnnx_sqrt(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Sqrt", input, NULL, output, ZDNNX_SQRT_OP);
  CHECK_ZDNN_STATUS(status, "zdnn_sqrt");
  return status;
}
