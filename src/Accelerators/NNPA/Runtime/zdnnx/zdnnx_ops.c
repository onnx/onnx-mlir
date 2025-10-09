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

#ifdef ZDNNX_WITH_OMP
#include "omp_ops.h"
#endif

#include "seq_ops.h"
#include "zdnnx_ops.h"
#include "zdnnx_ops_private.h"

static inline zdnn_status call_unary_op(const char *msg,
    const zdnn_ztensor *input, const void *scalar_input, zdnn_ztensor *output,
    ElemementwiseOp op_type) {
  zdnn_status status;
  ZDNNX_CALL_FUNC(msg, zdnnx_seq_unary_elementwise, zdnnx_omp_unary_elementwise,
      input, scalar_input, output, op_type);
  return status;
}

static inline zdnn_status call_binary_op(const char *msg,
    const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
    zdnn_ztensor *output, ElemementwiseOp op_type) {
  zdnn_status status;
  ZDNNX_CALL_FUNC(msg, zdnnx_seq_binary_elementwise,
      zdnnx_omp_binary_elementwise, input_a, input_b, output, op_type);
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
  ZDNNX_CALL_FUNC("MatMul", zdnnx_seq_matmul, zdnnx_omp_matmul, input_a,
      input_b, input_c, op_type, output, /*is_bcast=*/false);
  ZDNNX_CHECK_STATUS(status, "zdnn_matmul");
  return status;
}

zdnn_status zdnnx_matmul_bcast_op(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, int op_type,
    zdnn_ztensor *output) {
  zdnn_status status;
  ZDNNX_CALL_FUNC("MatMul", zdnnx_seq_matmul, zdnnx_omp_matmul, input_a,
      input_b, input_c, op_type, output, /*is_bcast=*/true);
  ZDNNX_CHECK_STATUS(status, "zdnn_matmul_bcast");
  return status;
}

zdnn_status zdnnx_matmul_transpose_op(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, int transpose_a,
    int transpose_b, int opType, zdnn_ztensor *output) {
  zdnn_status status;
  ZDNNX_CALL_FUNC("Transposed MatMul", zdnn_matmul_transpose_op,
      zdnn_matmul_transpose_op, input_a, input_b, input_c, transpose_a,
      transpose_b, opType, output);
  ZDNNX_CHECK_STATUS(status, "zdnn_matmul_transpose");
  return status;
}

zdnn_status zdnnx_quantized_matmul_op(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, const int8_t clip_min, const int8_t clip_max,
    const bool disable_clipping, const bool dequantize, const bool pre_computed,
    void *work_area, zdnn_ztensor *output) {
  zdnn_status status;
  ZDNNX_CALL_FUNC("Quantized MatMul", zdnn_quantized_matmul_op,
      zdnnx_omp_quantized_matmul, input_a, input_b, input_c, op_type, clip_min,
      clip_max, disable_clipping, dequantize, pre_computed, work_area, output);
  ZDNNX_CHECK_STATUS(status, "zdnn_quantized_matmul_op");
  return status;
}

zdnn_status zdnnx_add(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Add", inputA, inputB, output, ZDNNX_ADD_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_add");
  return status;
}

zdnn_status zdnnx_sub(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Sub", inputA, inputB, output, ZDNNX_SUB_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_sub");
  return status;
}

zdnn_status zdnnx_mul(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Mul", inputA, inputB, output, ZDNNX_MUL_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_mul");
  return status;
}

zdnn_status zdnnx_div(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Div", inputA, inputB, output, ZDNNX_DIV_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_div");
  return status;
}

zdnn_status zdnnx_min(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Min", inputA, inputB, output, ZDNNX_MIN_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_min");
  return status;
}

zdnn_status zdnnx_max(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_binary_op("Max", inputA, inputB, output, ZDNNX_MAX_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_max");
  return status;
}

zdnn_status zdnnx_exp(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status = call_unary_op("Exp", input, NULL, output, ZDNNX_EXP_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_exp");
  return status;
}

zdnn_status zdnnx_log(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status = call_unary_op("Log", input, NULL, output, ZDNNX_LOG_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_log");
  return status;
}

zdnn_status zdnnx_relu(const zdnn_ztensor *input, const void *clipping_value,
    zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Relu", input, clipping_value, output, ZDNNX_RELU_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_relu");
  return status;
}

zdnn_status zdnnx_sigmoid(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Sigmoid", input, NULL, output, ZDNNX_SIGMOID_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_sigmoid");
  return status;
}

zdnn_status zdnnx_tanh(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Tanh", input, NULL, output, ZDNNX_TANH_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_tanh");
  return status;
}

zdnn_status zdnnx_softmax(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output) {
  zdnn_status status;
  ZDNNX_CALL_FUNC("Softmax", zdnnx_seq_softmax, zdnnx_omp_softmax, input,
      save_area, act_func, output);
  ZDNNX_CHECK_STATUS(status, "zdnn_softmax");
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
  ZDNNX_CHECK_STATUS(status, "zdnn_gelu");
  return status;
}

zdnn_status zdnnx_invsqrt(
    const zdnn_ztensor *input, float epsilon, zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Invsqrt", input, &epsilon, output, ZDNNX_INVSQRT_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_invsqrt");
  return status;
}

zdnn_status zdnnx_leaky_relu(const zdnn_ztensor *input,
    const void *clipping_value, float adjustment_factor, zdnn_ztensor *output) {
  zdnn_status status;
  ZDNNX_CALL_FUNC("LeakyRelu", zdnn_leaky_relu, zdnn_leaky_relu, input,
      clipping_value, adjustment_factor, output);
  ZDNNX_CHECK_STATUS(status, "zdnn_leakyrelu");
  return status;
}

zdnn_status zdnnx_reduce(const zdnn_ztensor *input, void *save_area, int opType,
    zdnn_ztensor *output) {
  zdnn_status status;
  ZDNNX_CALL_FUNC(
      "Reduce", zdnn_reduce, zdnn_reduce, input, save_area, opType, output);
  ZDNNX_CHECK_STATUS(status, "zdnn_reduce");
  return status;
}

zdnn_status zdnnx_sqrt(const zdnn_ztensor *input, zdnn_ztensor *output) {
  zdnn_status status =
      call_unary_op("Sqrt", input, NULL, output, ZDNNX_SQRT_OP);
  ZDNNX_CHECK_STATUS(status, "zdnn_sqrt");
  return status;
}
