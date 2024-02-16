/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- NNPALimit.h ----------------------------------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// The NNPA constant values.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

// The NNPA maximum supported dimension index size value by using
// zdnn_get_nnpa_max_dim_idx_size() This value depends on HW.
static constexpr int64_t NNPA_MAXIMUM_DIMENSION_INDEX_SIZE = 32768;

// The NNPA maximum supported tensor size (in bytes)
// by using zdnn_get_nnpa_max_tensor_size()
// This value depends on HW.
static constexpr int64_t NNPA_MAXIMUM_TENSOR_SIZE = 4294967296;

// See zDNN API doc
static constexpr int64_t MAXIMUM_NUM_HIDDEN_SIZE_LSTM = 8192;
static constexpr int64_t MAXIMUM_NUM_HIDDEN_SIZE_GRU = 10880;

// The NNPA levels.
static constexpr const char *NNPA_Z16 = "z16";

// Maximum/Minimum value in dlfloat16.
// dlfloat value =  (-1)^s * 2^(e-31) * (1 + m/512), e=[0, 63], m=[0, 511],
// according to the paper: "DLFloat: A 16-b Floating Point Format Designed for
// Deep Learning Training and Inference", Ankur Agrawal, et al., (e=63, m=511)
// is preserved for NaN-Infinity, so use (s=0,e=63,m=510) as the maximum value
// and (s=1,e=63,m=510) as the minimum value.
static constexpr float DLF16_MAX = (1L << 32) * (1.0 + (510.0 / 512.0));
static constexpr float DLF16_MIN = -1 * (1L << 32) * (1.0 + (510.0 / 512.0));
