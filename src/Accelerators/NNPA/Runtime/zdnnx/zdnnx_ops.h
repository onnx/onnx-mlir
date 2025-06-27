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
extern uint32_t nnpa_max_dim_size_e4;
extern uint32_t nnpa_max_dim_size_e3;
extern uint32_t nnpa_max_dim_size_e2;
extern uint32_t nnpa_max_dim_size_e1;
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
// -----------------------------------------------------------------------------

zdnn_status zdnnx_matmul_op(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output);

zdnn_status zdnnx_matmul_bcast_op(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output);

// Elementwise Operations
zdnn_status zdnnx_add(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnnx_sub(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnnx_mul(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnnx_div(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnnx_min(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnnx_max(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnnx_exp(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnnx_log(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnnx_relu(
    const zdnn_ztensor *input, const void *clippingValue, zdnn_ztensor *output);
zdnn_status zdnnx_sigmoid(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnnx_softmax(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output);
zdnn_status zdnnx_tanh(const zdnn_ztensor *input, zdnn_ztensor *output);

// -----------------------------------------------------------------------------
// Extension Functions for arch15
// arch15 specific zdnn functions but with the `zdnnx` prefix.
// -----------------------------------------------------------------------------

zdnn_status zdnnx_gelu(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnnx_invsqrt(
    const zdnn_ztensor *input, float epsilon, zdnn_ztensor *output);
zdnn_status zdnnx_leaky_relu(const zdnn_ztensor *input,
    const void *clipping_value, float adjustment_factor, zdnn_ztensor *output);
zdnn_status zdnnx_sqrt(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnnx_matmul_transpose_op(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int transpose_a,
    int transpose_b, int opType, zdnn_ztensor *output);
zdnn_status zdnnx_reduce(const zdnn_ztensor *input, void *save_area,
    int op_type, zdnn_ztensor *output);

#endif // ZDNNX_ZDNNX_OPS_H
