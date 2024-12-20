/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- NNPALimit.hpp --------------------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// The NNPA constant values.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NNPA_LIMIT_H
#define ONNX_MLIR_NNPA_LIMIT_H

#include <stdint.h>
#include <string>

// Get maximum number of element for a given NNPA tensor. Dim is a tensor/memref
// index (from 0 to rank-1), with dim=0 being the outermost dimension and
// dim=(rank-1) being the innermost dimension. Return 0 if dimension is invalid.
// Generate assert if dim outside of rank, rank non-positive.
int64_t NNPAGetMaxForDim(int64_t dim, int64_t rank);

// The NNPA maximum supported tensor size (in bytes)
// by using zdnn_get_nnpa_max_tensor_size()
// This value depends on HW.
static constexpr int64_t NNPA_MAXIMUM_TENSOR_SIZE = 4294967296;

// See zDNN API doc
static constexpr int64_t MAXIMUM_NUM_HIDDEN_SIZE_LSTM = 8192;
static constexpr int64_t MAXIMUM_NUM_HIDDEN_SIZE_GRU = 10880;

// The NNPA levels. Newer versions must have larger numbers than older versions.
typedef enum NNPALevel {
  NONE = 0,
  M14 = 1, // Associated with march=arch14 | z16.
  M15 = 2, // Associated with march=arch15.
} NNPALevel;

// The NNPA ZDNN versions. Keep in sync with enum NNPALevel.
static constexpr uint64_t NNPA_ZDNN_VERSIONS[3] = {
    /*NONE*/ 0x0, /*M14*/ 0x010001, /*M15*/ 0x010101};

// Scan to NNPALevel and print from NNPALevel.
NNPALevel getNNPAFromFlags();
std::string getNNPAString(NNPALevel level);

/// A function to check whether the input NNPA level, ie. "z16" or "arch14", is
/// compatible with the current NNPA level.
bool isCompatibleWithNNPALevel(NNPALevel level);

/// A function to check whether the current --march (or deprecated --mcpu), ie.
/// "z16" or "arch14", is less than or equal to the given NNPA level.
bool isLessEqualNNPALevel(NNPALevel level);

// Maximum/Minimum value in dlfloat16.
// dlfloat value =  (-1)^s * 2^(e-31) * (1 + m/512), e=[0, 63], m=[0, 511],
// according to the paper: "DLFloat: A 16-b Floating Point Format Designed for
// Deep Learning Training and Inference", Ankur Agrawal, et al., (e=63, m=511)
// is preserved for NaN-Infinity, so use (s=0,e=63,m=510) as the maximum value
// and (s=1,e=63,m=510) as the minimum value.
static constexpr float DLF16_MAX = (1L << 32) * (1.0 + (510.0 / 512.0));
static constexpr float DLF16_MIN = -1 * (1L << 32) * (1.0 + (510.0 / 512.0));

#endif
