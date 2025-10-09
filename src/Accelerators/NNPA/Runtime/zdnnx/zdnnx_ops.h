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
zdnn_status zdnnx_quantized_matmul_op(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, const int8_t clip_min, const int8_t clip_max,
    const bool disable_clipping, const bool dequantize, const bool pre_computed,
    void *work_area, zdnn_ztensor *output);
zdnn_status zdnnx_reduce(const zdnn_ztensor *input, void *save_area,
    int op_type, zdnn_ztensor *output);

#endif // ZDNNX_ZDNNX_OPS_H
