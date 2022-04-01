/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- convert.hpp - Data Conversion --------------------------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains conversions for floating point to ZDNN formats
//
//===----------------------------------------------------------------------===//

#pragma once

#include <inttypes.h>

// Functions to convert data format.
uint64_t fp32_to_dlf16(
    float *input_data, uint16_t *output_data, uint64_t nbr_fields_to_convert);
uint64_t dlf16_to_fp32(
    uint16_t *input_data, float *output_data, uint64_t nbr_fields_to_convert);
