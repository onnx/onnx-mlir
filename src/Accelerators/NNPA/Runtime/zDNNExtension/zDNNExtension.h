/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ zDNNExtension.hpp ---------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// Sets of extensions to the zdnn library.
//
//===----------------------------------------------------------------------===//

#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

zdnn_status zdnn_matmul_op_ext(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, zdnn_ztensor *output);

#ifdef __cplusplus
}
#endif
