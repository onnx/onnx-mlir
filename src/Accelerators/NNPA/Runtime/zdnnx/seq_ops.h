/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- seq_ops.h ---------------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Sequential operations that split ztensors into tiles but use a single zAIU to
// run with the tiles.
//
//===----------------------------------------------------------------------===//

#ifndef ZDNNX_SEQ_OPS_H
#define ZDNNX_SEQ_OPS_H

#include "zdnnx_ops_private.h"

zdnn_status zdnnx_seq_unary_elementwise(const zdnn_ztensor *input,
    const void *scalar_input, zdnn_ztensor *output, ElemementwiseOp op_type);

zdnn_status zdnnx_seq_binary_elementwise(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, zdnn_ztensor *output, ElemementwiseOp op_type);

zdnn_status zdnnx_seq_softmax(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output);

zdnn_status zdnnx_seq_matmul(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, int op_type,
    zdnn_ztensor *output, bool is_bcast);

#endif // ZDNNX_SEQ_OPS_H
