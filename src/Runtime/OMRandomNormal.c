/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ OMRandomNormal.c - OMRandomNormal C Implementation -----------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C implementation of the OMRandomNormal functions.
//
//===----------------------------------------------------------------------===//

#include <math.h>
#include <stdlib.h>

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
