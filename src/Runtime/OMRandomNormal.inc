/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ OMRandomNormal.inc - OMRandomNormal C/C++ Implementation ------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ implementation of the OMRandomNormal functions.
//
//===----------------------------------------------------------------------===//


#ifdef __cplusplus

#include <random>

void get_random_normal_value_f64(
    double *result, int64_t size, double mean, double scale, double seed) {
  std::default_random_engine generator;
  generator.seed(seed);
  std::normal_distribution<double> distribution(mean, scale);
  for (int64_t index = 0; index < size; ++index)
    result[index] = distribution(generator);
}

void get_random_normal_value_f32(
    float *result, int64_t size, float mean, float scale, float seed) {
  std::default_random_engine generator;
  generator.seed(seed);
  std::normal_distribution<float> distribution(mean, scale);
  for (int64_t index = 0; index < size; ++index)
    result[index] = distribution(generator);
}

#else

#include <stdlib.h>
#include <math.h>

double uniformRandom() {
  return ((double)(rand()) + 1.0) / ((double)(RAND_MAX) + 1.0);
}

double normalRandom() {
  double random_1 = uniformRandom();
  double random_2 = uniformRandom();
  return cos(2 * 3.14159 * random_2) * sqrt(-2.0 * log(random_1));
}

void get_random_normal_value_f64(
    double *result, long long size, double mean, double scale, double seed) {
  for (long long index = 0; index < size; ++index)
    result[index] = normalRandom() * scale + mean;
}

void get_random_normal_value_f32(
    float *result, long long size, float mean, float scale, float seed) {
  srand(seed);
  for (long long index = 0; index < size; ++index)
    result[index] = normalRandom() * scale + mean;
}

#endif
