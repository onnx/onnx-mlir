/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- convert.cpp - Data Conversion --------------------------------===//
//
// Copyright 2020-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains conversions between floating point and ZDNN formats.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Support/Stickify/Convert.hpp"
#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Accelerators/NNPA/Support/Stickify/DLF16Conversion.hpp"

/// fp32 -> dlf16 conversion.
uint64_t fp32_to_dlf16(
    float *fp32_data, uint16_t *dflt16_data, uint64_t num_fields) {
  // Saturation limit: DLF16_MAX unless --nnpa-saturation asks for less.
  const float maxValue = onnx_mlir::getNNPASaturationLimit();
  const float minValue = -maxValue;
  uint64_t numBeyondRange = 0, numClippedByLimit = 0;
  for (uint64_t i = 0; i < num_fields; i++) {
    // Clip the fp32 value into the dlf16 range.
    float fp32_elem = fp32_data[i];
    if (fp32_elem < minValue || fp32_elem > maxValue) {
      if (fp32_elem < DLF16_MIN || fp32_elem > DLF16_MAX)
        numBeyondRange++;
      else
        numClippedByLimit++;
      fp32_elem = fp32_elem < minValue ? minValue : maxValue;
    }
    // Convert to dlf16.
    dflt16_data[i] = NNP1(fp32_elem).uint();
  }
  onnx_mlir::recordNNPAConstantSaturation(
      num_fields, numBeyondRange, numClippedByLimit);
  return num_fields;
}

/// dlf16 -> fp32 conversion.
uint64_t dlf16_to_fp32(
    uint16_t *dflt16_data, float *fp32_data, uint64_t num_fields) {
  for (uint64_t i = 0; i < num_fields; i++)
    fp32_data[i] = NNP1(dflt16_data[i]).convert();
  return num_fields;
}
