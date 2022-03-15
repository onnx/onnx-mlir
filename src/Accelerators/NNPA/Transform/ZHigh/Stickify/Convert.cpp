/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- convert.cpp - Data Conversion --------------------------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains conversions between floating point and ZDNN formats.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Transform/ZHigh/Stickify/Convert.hpp"
#include "src/Accelerators/NNPA/Transform/ZHigh/Stickify/DLF16Conversion.hpp"

/// fp32 -> dlf16 conversion.
uint64_t fp32_to_dlf16(
    float *fp32_data, uint16_t *dflt16_data, uint64_t num_fields) {
  for (uint64_t i = 0; i < num_fields; i++)
    dflt16_data[i] = NNP1(fp32_data[i]).uint();
  return num_fields;
}

/// dlf16 -> fp32 conversion.
uint64_t dlf16_to_fp32(
    uint16_t *dflt16_data, float *fp32_data, uint64_t num_fields) {
  for (uint64_t i = 0; i < num_fields; i++)
    fp32_data[i] = NNP1(dflt16_data[i]).convert();
  return num_fields;
}
