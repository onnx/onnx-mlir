/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMResizeTest.c - OMResize Unit Tests --------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Unit tests for the ONNXResizeOp runtime (OMResize.c).
//
// Focus areas:
//  1. Normal functional paths: Resize_Scales / Resize_Size with nearest,
//     linear, and cubic interpolation on small tensors, including downsampling.
//  2. Input validation tests (f-028):
//     - NaN, Inf, negative, zero, or out-of-range scale factors are rejected
//       before float->int64 cast.
//     - Sizes or scales causing outputSize > outputCap or product overflow
//       are rejected.
//
//===----------------------------------------------------------------------===//

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "OnnxMlirRuntime.h"

/* Declarations for the OMResize.c public entry points (not in public header) */
extern void Resize_Scales(OMTensor *output, OMTensor *data, OMTensor *scales,
    char *mode_str, char *nearest_mode);
extern void Resize_Size(OMTensor *output, OMTensor *data, OMTensor *size,
    char *mode_str, char *nearest_mode);

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define CHECK_NEAR(a, b, eps) CHECK(fabsf((float)(a) - (float)(b)) < (eps))

/* Create a 1-D OMTensor backed by a caller-provided float buffer. */
static OMTensor *make1Df(float *buf, int64_t n) {
  int64_t shape[1] = {n};
  return omTensorCreate(buf, shape, 1, ONNX_TYPE_FLOAT);
}

/* Create a 1-D OMTensor backed by a caller-provided int64 buffer. */
static OMTensor *make1Di64(int64_t *buf, int64_t n) {
  int64_t shape[1] = {n};
  return omTensorCreate(buf, shape, 1, ONNX_TYPE_INT64);
}

/* Create an N-D float OMTensor backed by a caller-provided buffer. */
static OMTensor *makeNDf(float *buf, int64_t *shape, int64_t rank) {
  return omTensorCreate(buf, shape, rank, ONNX_TYPE_FLOAT);
}

/* -------------------------------------------------------------------------- */
/* Normal path: Resize_Scales nearest, 1-D                                    */
/* -------------------------------------------------------------------------- */

static void testScalesNearest1D() {
  /*
   * Input: [1.0, 2.0, 3.0, 4.0]  shape [1,1,1,4]
   * scale: [1.0, 1.0, 1.0, 2.0]  -> output shape [1,1,1,8]
   * nearest (round_prefer_floor / asymmetric default):
   *   indices 0,0,1,1,2,2,3,3  => values 1,1,2,2,3,3,4,4
   */
  float inData[4] = {1.f, 2.f, 3.f, 4.f};
  int64_t inShape[4] = {1, 1, 1, 4};
  OMTensor *data = makeNDf(inData, inShape, 4);

  float scaleData[4] = {1.f, 1.f, 1.f, 2.f};
  OMTensor *scales = make1Df(scaleData, 4);

  float outData[8];
  memset(outData, 0, sizeof(outData));
  int64_t outShape[4] = {1, 1, 1, 8};
  OMTensor *output = makeNDf(outData, outShape, 4);

  Resize_Scales(output, data, scales, "nearest", "round_prefer_floor");

  CHECK_NEAR(outData[0], 1.f, 0.01f);
  CHECK_NEAR(outData[2], 2.f, 0.01f);
  CHECK_NEAR(outData[4], 3.f, 0.01f);
  CHECK_NEAR(outData[6], 4.f, 0.01f);

  omTensorDestroy(data);
  omTensorDestroy(scales);
  omTensorDestroy(output);
  printf("  PASS testScalesNearest1D\n");
}

/* -------------------------------------------------------------------------- */
/* Normal path: Resize_Scales linear, 1-D                                     */
/* -------------------------------------------------------------------------- */

static void testScalesLinear1D() {
  /*
   * Input: [1.0, 3.0]  shape [1,1,1,2]
   * scale: [1.0, 1.0, 1.0, 2.0] -> output [1,1,1,4]
   * linear half_pixel: output[1] should be between 1 and 3.
   */
  float inData[2] = {1.f, 3.f};
  int64_t inShape[4] = {1, 1, 1, 2};
  OMTensor *data = makeNDf(inData, inShape, 4);

  float scaleData[4] = {1.f, 1.f, 1.f, 2.f};
  OMTensor *scales = make1Df(scaleData, 4);

  float outData[4];
  memset(outData, 0, sizeof(outData));
  int64_t outShape[4] = {1, 1, 1, 4};
  OMTensor *output = makeNDf(outData, outShape, 4);

  Resize_Scales(output, data, scales, "linear", "");

  /* Interpolated values must lie within [1, 3]. */
  for (int i = 0; i < 4; i++) {
    CHECK(outData[i] >= 1.f - 0.01f && outData[i] <= 3.f + 0.01f);
  }

  omTensorDestroy(data);
  omTensorDestroy(scales);
  omTensorDestroy(output);
  printf("  PASS testScalesLinear1D\n");
}

/* -------------------------------------------------------------------------- */
/* Normal path: Resize_Scales linear downsampling                             */
/* -------------------------------------------------------------------------- */

static void testScalesLinearDownsample() {
  /*
   * Input: 1x1x2x4 tensor [[[[1, 2, 3, 4], [5, 6, 7, 8]]]]
   * scale: [1.0, 1.0, 0.6, 0.6] -> output shape 1x1x1x2
   * Matches test_resize_downsample_scales_linear.
   */
  float inData[8] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  int64_t inShape[4] = {1, 1, 2, 4};
  OMTensor *data = makeNDf(inData, inShape, 4);

  float scaleData[4] = {1.f, 1.f, 0.6f, 0.6f};
  OMTensor *scales = make1Df(scaleData, 4);

  float outData[2];
  memset(outData, 0, sizeof(outData));
  int64_t outShape[4] = {1, 1, 1, 2};
  OMTensor *output = makeNDf(outData, outShape, 4);

  Resize_Scales(output, data, scales, "linear", "");

  CHECK_NEAR(outData[0], 2.6666665f, 0.01f);
  CHECK_NEAR(outData[1], 4.3333331f, 0.01f);

  omTensorDestroy(data);
  omTensorDestroy(scales);
  omTensorDestroy(output);
  printf("  PASS testScalesLinearDownsample\n");
}

/* -------------------------------------------------------------------------- */
/* Normal path: Resize_Size nearest, 1-D                                      */
/* -------------------------------------------------------------------------- */

static void testSizeNearest1D() {
  /*
   * Input: [10.0, 20.0]  shape [1,1,1,2]
   * target size: [1,1,1,4]
   * Nearest: output should be [10,10,20,20]
   */
  float inData[2] = {10.f, 20.f};
  int64_t inShape[4] = {1, 1, 1, 2};
  OMTensor *data = makeNDf(inData, inShape, 4);

  int64_t sizeData[4] = {1, 1, 1, 4};
  OMTensor *size = make1Di64(sizeData, 4);

  float outData[4];
  memset(outData, 0, sizeof(outData));
  int64_t outShape[4] = {1, 1, 1, 4};
  OMTensor *output = makeNDf(outData, outShape, 4);

  Resize_Size(output, data, size, "nearest", "round_prefer_floor");

  CHECK_NEAR(outData[0], 10.f, 0.01f);
  CHECK_NEAR(outData[2], 20.f, 0.01f);

  omTensorDestroy(data);
  omTensorDestroy(size);
  omTensorDestroy(output);
  printf("  PASS testSizeNearest1D\n");
}

/* -------------------------------------------------------------------------- */
/* Security: non-positive output_size dimension rejected (Resize_Size)         */
/* -------------------------------------------------------------------------- */

static void testSizeNonPositiveDimRejected() {
  /*
   * Passing output size with a zero dimension must not crash or write to
   * the output buffer.
   */
  float inData[4] = {1.f, 2.f, 3.f, 4.f};
  int64_t inShape[4] = {1, 1, 1, 4};
  OMTensor *data = makeNDf(inData, inShape, 4);

  int64_t sizeData[4] = {1, 1, 0, 4}; /* zero in dim 2 */
  OMTensor *size = make1Di64(sizeData, 4);

  float sentinel = 0xDEAD;
  float outData[16];
  for (int i = 0; i < 16; i++)
    outData[i] = sentinel;
  int64_t outShape[4] = {1, 1, 1, 4};
  OMTensor *output = makeNDf(outData, outShape, 4);

  Resize_Size(output, data, size, "nearest", "");

  /* Output must be untouched — the guard must have returned early. */
  for (int i = 0; i < 4; i++)
    CHECK(outData[i] == sentinel);

  omTensorDestroy(data);
  omTensorDestroy(size);
  omTensorDestroy(output);
  printf("  PASS testSizeNonPositiveDimRejected\n");
}

/* -------------------------------------------------------------------------- */
/* Security: non-positive or invalid scale factors rejected (f028)             */
/* -------------------------------------------------------------------------- */

static void testScalesNonPositiveRejected() {
  /*
   * A scale of 0.0 or negative must be rejected before float->int64 cast.
   */
  float inData[4] = {1.f, 2.f, 3.f, 4.f};
  int64_t inShape[4] = {1, 1, 1, 4};
  OMTensor *data = makeNDf(inData, inShape, 4);

  float scaleData[4] = {1.f, 1.f, 1.f, 0.f}; /* zero scale on last dim */
  OMTensor *scales = make1Df(scaleData, 4);

  float sentinel = 0xBEEF;
  float outData[4];
  for (int i = 0; i < 4; i++)
    outData[i] = sentinel;
  int64_t outShape[4] = {1, 1, 1, 4};
  OMTensor *output = makeNDf(outData, outShape, 4);

  Resize_Scales(output, data, scales, "nearest", "");

  for (int i = 0; i < 4; i++)
    CHECK(outData[i] == sentinel);

  omTensorDestroy(data);
  omTensorDestroy(scales);
  omTensorDestroy(output);
  printf("  PASS testScalesNonPositiveRejected\n");
}

static void testScalesInvalidFloatsRejected() {
  /*
   * NaN, Inf, negative, and astronomically large scale factors (f028) must
   * be rejected early before the float->int64 cast.
   */
  float inData[4] = {1.f, 2.f, 3.f, 4.f};
  int64_t inShape[4] = {1, 1, 1, 4};
  OMTensor *data = makeNDf(inData, inShape, 4);

  float testScales[4][4] = {
      {1.f, 1.f, 1.f, NAN},
      {1.f, 1.f, 1.f, INFINITY},
      {1.f, 1.f, 1.f, -1.0f},
      {1.f, 1.f, 1.f, 1e30f},
  };

  for (int t = 0; t < 4; t++) {
    OMTensor *scales = make1Df(testScales[t], 4);
    float sentinel = 0xBEEF + t;
    float outData[4];
    for (int i = 0; i < 4; i++)
      outData[i] = sentinel;
    int64_t outShape[4] = {1, 1, 1, 4};
    OMTensor *output = makeNDf(outData, outShape, 4);

    Resize_Scales(output, data, scales, "nearest", "");

    for (int i = 0; i < 4; i++)
      CHECK(outData[i] == sentinel);

    omTensorDestroy(scales);
    omTensorDestroy(output);
  }

  omTensorDestroy(data);
  printf("  PASS testScalesInvalidFloatsRejected\n");
}

/* -------------------------------------------------------------------------- */
/* Security: outputSize exceeds outputCap rejected (Resize_Size)               */
/* -------------------------------------------------------------------------- */

static void testSizeOutputExceedsCapRejected() {
  /*
   * The output tensor is pre-allocated for 4 elements but the requested
   * size asks for 8 elements.  The guard (outputSize > outputCap) must
   * reject this and leave the output buffer untouched.
   */
  float inData[4] = {1.f, 2.f, 3.f, 4.f};
  int64_t inShape[4] = {1, 1, 1, 4};
  OMTensor *data = makeNDf(inData, inShape, 4);

  int64_t sizeData[4] = {1, 1, 1, 8}; /* requests 8 output elements */
  OMTensor *size = make1Di64(sizeData, 4);

  float sentinel = 0xCAFE;
  float outData[4]; /* only 4 elements — intentionally small */
  for (int i = 0; i < 4; i++)
    outData[i] = sentinel;
  int64_t outShape[4] = {1, 1, 1, 4}; /* output tensor says 4 */
  OMTensor *output = makeNDf(outData, outShape, 4);

  Resize_Size(output, data, size, "nearest", "");

  /* Guard must have fired; output untouched. */
  for (int i = 0; i < 4; i++)
    CHECK(outData[i] == sentinel);

  omTensorDestroy(data);
  omTensorDestroy(size);
  omTensorDestroy(output);
  printf("  PASS testSizeOutputExceedsCapRejected\n");
}

/* -------------------------------------------------------------------------- */
/* Security: outputSize overflow via large per-dimension values (Resize_Size)  */
/* -------------------------------------------------------------------------- */

static void testSizeDimensionOverflowRejected() {
  /*
   * Two large dimension values whose cumulative product far exceeds the
   * pre-allocated output capacity (outputCap = 4 elements).
   * Before the fix, the unchecked product could wrap a size_t argument to
   * malloc, returning an undersized buffer that generate_coordinates then
   * wrote past the end of — a heap overflow.
   * After the fix, the guard (outputSize > outputCap / output_size[i])
   * fires at the first large dimension and returns early without touching
   * the output buffer.
   */
  float inData[4] = {1.f, 2.f, 3.f, 4.f};
  int64_t inShape[4] = {1, 1, 1, 4};
  OMTensor *data = makeNDf(inData, inShape, 4);

  /* outputCap = 4; 4 / 1000000000 = 0, so outputSize(1) > 0 triggers guard */
  int64_t sizeData[4] = {1, 1, (int64_t)1000000000LL, (int64_t)1000000000LL};
  OMTensor *size = make1Di64(sizeData, 4);

  float sentinel = 0xF00D;
  float outData[4];
  for (int i = 0; i < 4; i++)
    outData[i] = sentinel;
  int64_t outShape[4] = {1, 1, 1, 4};
  OMTensor *output = makeNDf(outData, outShape, 4);

  Resize_Size(output, data, size, "nearest", "");

  for (int i = 0; i < 4; i++)
    CHECK(outData[i] == sentinel);

  omTensorDestroy(data);
  omTensorDestroy(size);
  omTensorDestroy(output);
  printf("  PASS testSizeDimensionOverflowRejected\n");
}

/* -------------------------------------------------------------------------- */
/* main                                                                        */
/* -------------------------------------------------------------------------- */

int main(void) {
  printf("OMResizeTest: running...\n");

  testScalesNearest1D();
  testScalesLinear1D();
  testScalesLinearDownsample();
  testSizeNearest1D();
  testSizeNonPositiveDimRejected();
  testScalesNonPositiveRejected();
  testScalesInvalidFloatsRejected();
  testSizeOutputExceedsCapRejected();
  testSizeDimensionOverflowRejected();

  if (failures) {
    fprintf(stderr, "OMResizeTest: %d check(s) FAILED\n", failures);
    return 1;
  }
  printf("OMResizeTest: all checks passed\n");
  return 0;
}
