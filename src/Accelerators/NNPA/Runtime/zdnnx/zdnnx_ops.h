/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- zdnnx_ops.h -------------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Operations in the zdnn extension to replace zdnn operations.
//
//===----------------------------------------------------------------------===//

#ifndef ZDNNX_ZDNNX_OPS_H
#define ZDNNX_ZDNNX_OPS_H

#include "zdnn.h"
#include "zdnnx.h"

// Keep these values to avoid calling zdnn functions multiple times.
extern uint32_t nnpa_max_dim_size;
extern uint64_t nnpa_max_tensor_size;

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

// Get the max number of elements per dim on zAIU.
uint32_t get_nnpa_max_dim_size(zdnnx_axis dim_index);

// Get the max number of elements per tensor on zAIU.
uint64_t get_nnpa_max_tensor_size();

/**
 * \brief Check zdnn status
 *
 * Check if the zdnn status is not a zdnn_ok and print out the
 * status message along with the error
 *
 * @param status zdnn status
 * @param zdnn_name name of the zdnn api
 */
void check_status(zdnn_status status, const char *zdnn_name);

#define CHECK_ZDNN_STATUS(status, zdnn_name) check_status((status), (zdnn_name))

// -----------------------------------------------------------------------------
// Extension Functions
// Same name as zdnn functions but with the `zdnnx` prefix instead of `zdnn`.
//
// TODO: change function names to zdnnx_. Keep the current names for
// a while since onnx-mlir is calling them.
// -----------------------------------------------------------------------------

zdnn_status zdnn_matmul_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output);

zdnn_status zdnn_matmul_bcast_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output);

// Elementwise Operations
zdnn_status zdnn_add_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_sub_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_mul_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_div_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_min_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_max_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_exp_ext(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_log_ext(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_relu_ext(
    const zdnn_ztensor *input, const void *clippingValue, zdnn_ztensor *output);
zdnn_status zdnn_sigmoid_ext(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_softmax_ext(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output);
zdnn_status zdnn_tanh_ext(const zdnn_ztensor *input, zdnn_ztensor *output);

// -----------------------------------------------------------------------------
// Extension Functions for arch15
// arch15 specific zdnn functions but with the `_ext` postfix.
// -----------------------------------------------------------------------------

zdnn_status zdnn_gelu_ext(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_invsqrt_ext(
    const zdnn_ztensor *input, float epsilon, zdnn_ztensor *output);
zdnn_status zdnn_leaky_relu_ext(const zdnn_ztensor *input,
    const void *clipping_value, float adjustment_factor, zdnn_ztensor *output);
zdnn_status zdnn_sqrt_ext(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_matmul_transpose_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int transpose_a,
    int transpose_b, int opType, zdnn_ztensor *output);
zdnn_status zdnn_reduce_ext(const zdnn_ztensor *input, void *save_area,
    int op_type, zdnn_ztensor *output);

#endif // ZDNNX_ZDNNX_OPS_H
