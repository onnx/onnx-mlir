/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----OMRandomUniform.inc - OMRandomUniform C/C++ Implementation//------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementation of the OMRandomUniform functions.
//
//===----------------------------------------------------------------------===//

#include "onnx-mlir/Runtime/OMTensor.h"
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
#include <random>
#endif

//===----------------------------------------------------------------------===//
// Float32 version
//===----------------------------------------------------------------------===//
OMTensor *run_uniform_random_f32(
    OMTensor *output_tensor, float low, float high, float seed) {
  float *output_ptr = (float *)omTensorGetDataPtr(output_tensor);
  int64_t num_elements = omTensorGetNumElems(output_tensor);

#ifdef __cplusplus
  std::default_random_engine generator;
  generator.seed(static_cast<unsigned int>(seed));
  std::uniform_real_distribution<float> distribution(low, high);
  for (int64_t i = 0; i < num_elements; ++i)
    output_ptr[i] = distribution(generator);
#else
  srand((unsigned int)seed);
  for (int64_t i = 0; i < num_elements; ++i) {
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    output_ptr[i] = low + r * (high - low);
  }
#endif

  return output_tensor;
}

//===----------------------------------------------------------------------===//
// Float64 version
//===----------------------------------------------------------------------===//
OMTensor *run_uniform_random_f64(
    OMTensor *output_tensor, double low, double high, double seed) {
  double *output_ptr = (double *)omTensorGetDataPtr(output_tensor);
  int64_t num_elements = omTensorGetNumElems(output_tensor);

#ifdef __cplusplus
  std::default_random_engine generator;
  generator.seed(static_cast<unsigned int>(seed));
  std::uniform_real_distribution<double> distribution(low, high);
  for (int64_t i = 0; i < num_elements; ++i)
    output_ptr[i] = distribution(generator);
#else
  srand((unsigned int)seed);
  for (int64_t i = 0; i < num_elements; ++i) {
    double r = (double)rand() / ((double)RAND_MAX + 1.0);
    output_ptr[i] = low + r * (high - low);
  }
#endif

  return output_tensor;
}

//===----------------------------------------------------------------------===//
// Dispatch based on tensor data type
//===----------------------------------------------------------------------===//
OMTensor *run_uniform_random(
    OMTensor *output_tensor, float low, float high, float seed) {
  OM_DATA_TYPE dtype = omTensorGetDataType(output_tensor);
  switch (dtype) {
  case ONNX_TYPE_FLOAT:
    return run_uniform_random_f32(output_tensor, low, high, seed);
  case ONNX_TYPE_DOUBLE:
    return run_uniform_random_f64(
        output_tensor, (double)low, (double)high, (double)seed);
  default:
    return NULL;
  }
}
