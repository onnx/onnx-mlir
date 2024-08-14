/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OMResize.cpp - OMTensor C/C++ Implementation ----------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ neutral implementation of OMTensorList data
// structures and helper functions.
//
//===----------------------------------------------------------------------===//
#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"

/**
 * The runtime implementation of ONNXResizeOp is manually translated python
 * implementation: onnx/onnx/backend/test/case/node/resize.py (onnx v1.9.0)
 * This implementation can be improved in efficiency.
 **/

static void linear_coeffs(float ratio, float coeffs_buffer[2], int mode) {
  coeffs_buffer[0] = 1 - ratio;
  coeffs_buffer[1] = ratio;
}

static void nearest_coeffs(float ratio, float coeffs_buffer[2], int mode) {
  /* integer ratio is handled outside */
  switch (mode) {
  case 0: // round_prefer_float
    coeffs_buffer[0] = ratio <= 0.5 ? 1 : 0;
    coeffs_buffer[1] = ratio > 0.5 ? 1 : 0;
    break;
  case 1: // round_prefer_ceil
    coeffs_buffer[0] = ratio < 0.5 ? 1 : 0;
    coeffs_buffer[1] = ratio >= 0.5 ? 1: 0;
    break;
  case 2: // floor
    coeffs_buffer[0] = 1;
    coeffs_buffer[1] = 0;
    break;
  case 4: // ceil
    coeffs_buffer[0] = 0;
    coeffs_buffer[1] = 1;
    break;
  }
}

static void cubic_coeffs(float ratio, float coeffs_buffer[4], int mode) {
  float A = -0.75;
  // A may have different value for different coordinate_transformation_mode
  // Currently, only default mode is supported
  coeffs_buffer[0] =
      ((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) - 4 * A;
  coeffs_buffer[1] = ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1;
  coeffs_buffer[2] =
      ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1;
  coeffs_buffer[3] =
      ((A * ((1 - ratio) + 1) - 5 * A) * ((1 - ratio) + 1) + 8 * A) *
          ((1 - ratio) + 1) -
      4 * A;
}

static void get_neighbor(float x, int64_t n, int limit, float *data,
    float *points, int exclude_outside) {

  // inline the python get_neighbor_idx function without real padding
  // Avoid malloc/free
  // nearest indx: identify the central idx first, then select from both side
  // If the central idx is right to x (>= x), favor the left one
  // == from the example

  int pad_width = ceil(n / 2);
  x += pad_width;
  float r = x - floor(x);

  int start, end;
  int c;
  if (r > 0.5) {
    c = (int)(floor(x)) + 1;
  } else {
    c = (int)(floor(x));
  }

  int rest = n - 1;
  int half = rest / 2;
  if (rest == 0) {
    start = end = c;
  } else if (rest % 2 == 0) {
    start = c - half;
    end = c + half;
  } else if (r == 0) {
    start = c - half - 1;
    end = c + half;
  } else {
    if (r > 0.5) {
      end = c + half;
      start = c - half - 1;
    } else {
      end = c + half + 1;
      start = c - half;
    }
  }

  start -= pad_width;
  end -= pad_width;

  for (int i = start; i <= end; i++) {
    if (i < 0) {
      if (exclude_outside)
        points[i - start] = 0;
      else
        points[i - start] = data[0];
    } else if (i >= limit) {
      if (exclude_outside)
        points[i - start] = 0;
      else
        points[i - start] = data[limit - 1];
    } else {
      points[i - start] = data[i];
    }
  }
}

typedef void (*Coeff_Func_t)(float, float *, int mode);

static float interpolate_1d_with_x(OMTensor *data, float scale_factor, float x,
    Coeff_Func_t get_coeffs, float *coeffs_buffer, int coeffs_n, float roi,
    float extrapolation_value, int coordinate_transformation_mode,
    int exclude_outside, int mode) {

  int64_t input_width = omTensorGetShape(data)[0];
  float x_ori;
  switch (coordinate_transformation_mode) {
  case 0: // half_pixel
    x_ori = (x + 0.5) / scale_factor - 0.5;
    break;
  case 1: // asymmetric
    x_ori = x / scale_factor;
    break;
  }
  int64_t x_ori_int = floor(x_ori);

  float ratio = x_ori - x_ori_int;
  if (ratio == 0)
    ratio = 1;

  get_coeffs(ratio, coeffs_buffer, mode);
  int64_t n = coeffs_n;

  // float points[coeffs_n];
  float *points = (float *)malloc(sizeof(float) * coeffs_n);
  assert(points && "failed to allocate memory for points");

  get_neighbor(x_ori, n, input_width, (float *)omTensorGetDataPtr(data), points,
      exclude_outside);
  float sum = 0.;
  for (int i = 0; i < n; i++) {
    sum += coeffs_buffer[i] * points[i];
  }
  free(points);
  // free OMTensor data1
  return sum;
}

static float interpolate_nd_with_x(OMTensor *data, int n, float *scale_factors,
    float *xs, Coeff_Func_t get_coeffs, float *coeffs_buffer, int coeffs_n,
    float roi, float extrapolation_value, int coordinate_transformation_mode,
    int exclude_outside, int mode) {
  if (n == 1) {
    return interpolate_1d_with_x(data, scale_factors[0], xs[0], get_coeffs,
        coeffs_buffer, coeffs_n, roi, extrapolation_value,
        coordinate_transformation_mode, exclude_outside, mode);
  } else {
    int64_t input_width = omTensorGetShape(data)[0];
    float *tempData = (float *)malloc(sizeof(float) * input_width);
    assert(tempData && "failed to allocate memory for tempData");
    int64_t tempShape[] = {input_width};

    int64_t stride = 1;
    for (int i = 1; i < n; i++) {
      stride *= omTensorGetShape(data)[i];
    }
    for (int i = 0; i < input_width; i++) {
      float *dataPtr = (float *)omTensorGetDataPtr(data) + i * stride;
      OMTensor *data1 = omTensorCreate(
          dataPtr, omTensorGetShape(data) + 1, n - 1, ONNX_TYPE_FLOAT);
      tempData[i] = interpolate_nd_with_x(data1, n - 1, scale_factors + 1,
          xs + 1, get_coeffs, coeffs_buffer, coeffs_n, roi, extrapolation_value,
          coordinate_transformation_mode, exclude_outside, mode);
      omTensorDestroy(data1);
    }
    OMTensor *tempT = omTensorCreate(tempData, tempShape, 1, ONNX_TYPE_FLOAT);
    float ret = interpolate_1d_with_x(tempT, scale_factors[0], xs[0],
        get_coeffs, coeffs_buffer, coeffs_n, roi, extrapolation_value,
        coordinate_transformation_mode, exclude_outside, mode);
    omTensorDestroy(tempT);
    free(tempData);
    return ret;
  }
}

static void coordinate_step(int64_t rank, int64_t *output_size,
    int64_t *allCoordinates, int64_t currentRank, int64_t *currentIter,
    int64_t *currentPosition) {
  for (int i = 0; i < output_size[currentRank]; i++) {
    if (currentRank == rank - 1) {
      for (int j = 0; j < currentRank; j++) {
        *(allCoordinates + (*currentPosition) * rank + j) = currentIter[j];
      }
      *(allCoordinates + (*currentPosition) * rank + currentRank) = i;
      (*currentPosition)++;
    } else {
      currentIter[currentRank] = i;
      coordinate_step(rank, output_size, allCoordinates, currentRank + 1,
          currentIter, currentPosition);
    }
  }
}

static void generate_coordinates(
    int64_t rank, int64_t *output_size, int64_t *allCoordinates) {
  int64_t position = 0;
  int64_t *currentIter = (int64_t *)malloc(sizeof(int64_t) * rank);
  assert(currentIter && "failed to allocate memory for currentIter");
  coordinate_step(rank, output_size, allCoordinates, 0, currentIter, &position);
  free(currentIter);
}

static void interpolate_nd_OMTensor(OMTensor *output_OMT, OMTensor *data,
    int64_t mode, OMTensor *output_size_OMT, OMTensor *scale_factor_OMT,
    Coeff_Func_t get_coeffs, int coeffs_n, OMTensor *roi,
    float *extrapolation_value, int coordinate_transformation_mode,
    int exclude_outside) {
  assert(omTensorGetDataType(data) == ONNX_TYPE_FLOAT &&
         "Resize runtime: only float type is supported currently");

  int64_t rank = omTensorGetRank(data);
  const int64_t *inputShape = omTensorGetShape(data);
  float *scale_factor = NULL;
  int64_t *output_size = NULL;
  if (scale_factor_OMT != NULL)
    scale_factor = (float *)omTensorGetDataPtr(scale_factor_OMT);
  if (output_size_OMT != NULL)
    output_size = (int64_t *)omTensorGetDataPtr(output_size_OMT);
  if (scale_factor == NULL) {
    scale_factor = (float *)malloc(sizeof(float) * rank);
    assert(scale_factor && "failed to allocate memory for scale_factor");
    for (int i = 0; i < rank; i++) {
      scale_factor[i] = ((float)output_size[i]) / inputShape[i];
    }
  } else {
    output_size = (int64_t *)malloc(sizeof(int64_t) * rank);
    assert(output_size && "failed to allocate memory for output_size");
    for (int i = 0; i < rank; i++) {
      output_size[i] = scale_factor[i] * inputShape[i];
    }
  }

  int64_t outputSize = 1;
  for (int i = 0; i < rank; i++) {
    outputSize *= output_size[i];
  }
  float *outputData = (float *)omTensorGetDataPtr(output_OMT);

  // int64_t allCoordinates[outputSize][rank];
  int64_t *allCoordinates =
      (int64_t *)malloc(outputSize * rank * sizeof(int64_t));
  assert(allCoordinates && "failed to allocate memory for allCoordinates");
  generate_coordinates(rank, output_size, allCoordinates);

  // float coeffs_buffer[coeffs_n]; // = {1.0, 0.};
  float *coeffs_buffer = (float *)malloc(sizeof(float) * coeffs_n);
  assert(coeffs_buffer && "failed to allocate memory for coeffs_buffer");

  for (int i = 0; i < outputSize; i++) {
    float *Xs = (float *)malloc(sizeof(float) * rank);
    assert(Xs && "failed to allocate memory for Xs");
    for (int j = 0; j < rank; j++) {
      Xs[j] = *(allCoordinates + i * rank + j);
    }
    float r = interpolate_nd_with_x(
        /*OMTensor */ data,
        /* n */ 4,
        /*float scale_factor*/ scale_factor,
        /*floats *x*/ Xs,
        /* Coeff_Func_t*/ get_coeffs,
        /*float */ coeffs_buffer,
        /*int coeffs_n*/ coeffs_n,
        /*float roi*/ 0.,
        /*float extrapolation_value*/ 0.,
        /*int coordinate_transformation_mode*/ 0,
        /*exclude */ 0,
        /*mode */ 0);
    outputData[i] = r;
    free(Xs);
  }
  if (output_size_OMT == NULL)
    free(output_size);
  if (scale_factor_OMT == NULL)
    free(scale_factor);
  free(allCoordinates);
  free(coeffs_buffer);
}

// The parameters that are not used are commented out.
// I, Tong Chen, forgot why only the default value for them are used.
// ToFix: add the support for other value if needed
void Resize_Scales(
    OMTensor *output, OMTensor *data, OMTensor *scales, char *mode_str,
    char *nearest_mode) {
  Coeff_Func_t coeffs_f = NULL;
  int coeffs_n = 0;
  if (strcmp(mode_str, "nearest") == 0) {
    coeffs_f = nearest_coeffs;
    coeffs_n = 2;
  } else if (strncmp(mode_str, "linear", 6) == 0) {
    coeffs_f = linear_coeffs;
    coeffs_n = 2;
  } else if (strcmp(mode_str, "cubic") == 0) {
    coeffs_f = cubic_coeffs;
    coeffs_n = 4;
  } else {
    assert(0 && "Resize runtime: unsupported mode");
  }
  
  interpolate_nd_OMTensor(
      /*OMTensor */ output,
      /*OMTensor */ data,
      /*mode*/ 0,
      /*OMTensor output size */ NULL,
      /*OMTensor scales */ scales,
      /* Coeff_Func_t*/ coeffs_f,
      /*int coeffs_n*/ coeffs_n,
      /* roi */ NULL,
      /*float * extrapolation_value*/ 0,
      /*int coordinate_transformation_mode*/ 0,
      /*exclude */ 0);
}

void Resize_Size(
    OMTensor *output, OMTensor *data, OMTensor *size, char *mode_str,
    char *nearest_mode) {
  Coeff_Func_t coeffs_f = NULL;
  int coeffs_n = 0;
  if (strcmp(mode_str, "nearest") == 0) {
    coeffs_f = nearest_coeffs;
    coeffs_n = 2;
  } else if (strncmp(mode_str, "linear", 6) == 0) {
    coeffs_f = linear_coeffs;
    coeffs_n = 2;
  } else if (strcmp(mode_str, "cubic") == 0) {
    coeffs_f = cubic_coeffs;
    coeffs_n = 4;
  } else {
    assert(0 && "Resize runtime: unsupported mode");
  }
  interpolate_nd_OMTensor(
      /*OMTensor */ output,
      /*OMTensor */ data,
      /*mode*/ 0,
      /*OMTensor output size */ size,
      /*OMTensor scales */ NULL,
      /* Coeff_Func_t*/ coeffs_f,
      /*int coeffs_n*/ coeffs_n,
      /* roi */ NULL,
      /*float * extrapolation_value*/ 0,
      /*int coordinate_transformation_mode*/ 0,
      /*exclude */ 0);
}
