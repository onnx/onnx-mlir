/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- omp_ops.h ---------------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Parallel operations that split ztensors into tiles and use OpenMP to run
// with the tiles on multiple zAIUs.
//
//===----------------------------------------------------------------------===//

#ifndef ZDNNX_OMP_OPS_H
#define ZDNNX_OMP_OPS_H

#ifdef ZDNNX_WITH_OMP

#include "zdnnx_ops.h"
#include "zdnnx_ops_private.h"

// The total number of processors.
uint32_t zdnnx_get_num_procs();

zdnn_status zdnnx_omp_matmul(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, int op_type,
    zdnn_ztensor *output, bool is_bcast);

zdnn_status zdnnx_omp_unary_elementwise(const zdnn_ztensor *input,
    const void *scalar_input, zdnn_ztensor *output, ElemementwiseOp op_type);

zdnn_status zdnnx_omp_binary_elementwise(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, zdnn_ztensor *output, ElemementwiseOp op_type);

zdnn_status zdnnx_omp_softmax(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output);

#endif // ZDNNX_WITH_OMP

#endif // ZDNNX_OMP_OPS_H
