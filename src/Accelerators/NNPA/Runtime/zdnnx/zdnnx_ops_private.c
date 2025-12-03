/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ zdnnx_ops_private.c -------------------------===//
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

#include <stdio.h>

#include "zdnnx_ops_private.h"

// Keep these values to avoid calling zdnn functions multiple times.
static uint32_t nnpa_max_dim_size_e4 = 0;
static uint32_t nnpa_max_dim_size_e3 = 0;
static uint32_t nnpa_max_dim_size_e2 = 0;
static uint32_t nnpa_max_dim_size_e1 = 0;
static uint64_t nnpa_max_tensor_size = 0;

uint32_t zdnnx_get_nnpa_max_dim_size(zdnnx_axis dim_index) {
  switch (dim_index) {
  case E4:
    if (nnpa_max_dim_size_e4 == 0)
      nnpa_max_dim_size_e4 = zdnnx_is_telum_1 ? zdnn_get_nnpa_max_dim_idx_size()
                                              : zdnn_get_max_for_dim(4);
    return nnpa_max_dim_size_e4;
  case E3:
    if (nnpa_max_dim_size_e3 == 0)
      nnpa_max_dim_size_e3 = zdnnx_is_telum_1 ? zdnn_get_nnpa_max_dim_idx_size()
                                              : zdnn_get_max_for_dim(3);
    return nnpa_max_dim_size_e3;
  case E2:
    if (nnpa_max_dim_size_e2 == 0)
      nnpa_max_dim_size_e2 = zdnnx_is_telum_1 ? zdnn_get_nnpa_max_dim_idx_size()
                                              : zdnn_get_max_for_dim(2);
    return nnpa_max_dim_size_e2;
  case E1:
    if (nnpa_max_dim_size_e1 == 0)
      nnpa_max_dim_size_e1 = zdnnx_is_telum_1 ? zdnn_get_nnpa_max_dim_idx_size()
                                              : zdnn_get_max_for_dim(1);
    return nnpa_max_dim_size_e1;
  default:
    return 0;
  }
}

uint64_t zdnnx_get_nnpa_max_tensor_size() {
  if (nnpa_max_tensor_size == 0) {
    // zdnn_get_nnpa_max_tensor_size() returns size in bytes.
    nnpa_max_tensor_size = zdnn_get_nnpa_max_tensor_size() / 2;
  }
  return nnpa_max_tensor_size;
}

void zdnnx_check_status(zdnn_status status, const char *zdnn_name) {
  if (zdnnx_status_message_enabled && status != ZDNN_OK) {
    fprintf(stdout, "[zdnnx] %s : %s\n", zdnn_name,
        zdnn_get_status_message(status));
  }
}
