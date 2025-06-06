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

#ifdef __cplusplus

#include <random>

void get_uniform_random_value_f64(
    double *result, int64_t size, double min, double max, double seed) {
  std::default_random_engine generator;
  generator.seed(seed);
  std::uniform_real_distribution<double> distribution(min, max);
  for (int64_t index = 0; index < size; ++index)
    result[index] = distribution(generator);
}

void get_uniform_random_value_f32(
    double *result, int64_t size, double min, double max, double seed) {
  std::default_random_engine generator;
  generator.seed(seed);
  std::uniform_real_distribution<double> distribution(min, max);
  for (int64_t index = 0; index < size; ++index)
    result[index] = distribution(generator);
}

#else

#include <math.h>
#include <stdlib.h>

double uniformRandom() {
  return ((double)(rand()) + 1.0) / ((double)(RAND_MAX) + 1.0);
}

void get_uniform_random_value_f64(
    double *output, int64_t size, double min, double max, double seed) {
  srand((unsigned int)seed);
  for (int64_t i = 0; i < size; ++i) {
    double r = uniformRandom();
    output[i] = min + (max - min) * r;
  }
}

void get_uniform_random_value_f32(
    float *output, int64_t size, float min, float max, float seed) {
  srand((unsigned int)seed);
  for (int64_t i = 0; i < size; ++i) {
    float r = uniformRandom();
    output[i] = min + (max - min) * r;
  }
}

#endif
