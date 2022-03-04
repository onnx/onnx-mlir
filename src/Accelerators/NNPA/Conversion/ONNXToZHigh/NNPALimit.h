/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- NNPALimit.cpp --------------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// The NNPA maximum supported dimension index size value by using
// zdnn_get_nnpa_max_dim_idx_size() This value depends on HW.
//
//===----------------------------------------------------------------------===//

// The NNPA maximum supported dimension index size value by using
// zdnn_get_nnpa_max_dim_idx_size() This value depends on HW.
#define DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE 32768

// The NNPA maximum supported tensor size (in bytes)
// by using zdnn_get_nnpa_max_tensor_size()
// This value depends on HW.
#define DLCPP_MAXIMUM_TENSOR_SIZE 4294967296

// See zDNN API doc
#define MAXIMUM_NUM_HIDDEN_SIZE_LSTM 8192
#define MAXIMUM_NUM_HIDDEN_SIZE_GRU 10880
