/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- TestStickifyRNN.cpp - Unit tests for f031 RNN dim1 overflow -------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Regression tests for f031: get_rnn_concatenated_dim1() integer overflow.
//
// The original code computed PADDED(val) * num_gates in uint32_t, which
// silently wraps when val is large (~1.07 billion for LSTM, ~1.43 billion
// for GRU).  The result is a truncated dim1 far smaller than the per-gate
// buffer size the subsequent memset derives from the un-concatenated shape,
// producing a heap overwrite.
//
// get_rnn_concatenated_dim1() is a file-scoped helper, so we test it
// indirectly through generate_transformed_desc_concatenated() which is the
// public wrapper that consumes its return value.  A truncated dim1 causes the
// concatenated tfrmd_desc to compute a buffer size smaller than the per-gate
// buffer -- we detect this by checking that the call returns ZDNN_INVALID_SHAPE
// rather than silently succeeding with a dangerously small size.
//
// Normal (non-overflow) cases verify that the concatenated descriptor is
// produced correctly for both LSTM (4 gates) and GRU (3 gates).
//
//===----------------------------------------------------------------------===//

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "build/Release/include/zdnn.h"
}

#include "src/Accelerators/NNPA/Support/Stickify/Stickify.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// AIU constants (same as in Stickify.cpp).
static const uint32_t kStickCells = 64; // AIU_2BYTE_CELLS_PER_STICK
static uint32_t padded(uint32_t v) {
  return ((v + kStickCells - 1) / kStickCells) * kStickCells;
}

// ---------------------------------------------------------------------------
// Normal LSTM: val=128, padded=128, dim1 = 128*4 = 512
// ---------------------------------------------------------------------------
static int testLSTMNormalCase() {
  zdnn_tensor_desc pre, tfrmd;
  memset(&pre, 0, sizeof(pre));
  memset(&tfrmd, 0, sizeof(tfrmd));

  set_info_pre_transformed_desc(&pre, ZDNN_2DS, ZDNN_DLFLOAT16, {1, 128});

  zdnn_concat_info info = RNN_TYPE_LSTM | PREV_LAYER_UNI | USAGE_BIASES;
  zdnn_status st = generate_transformed_desc_concatenated(&pre, info, &tfrmd);
  if (st != ZDNN_OK) {
    fprintf(stderr, "FAIL: LSTM normal case: expected ZDNN_OK, got %d\n", st);
    return 1;
  }
  uint32_t expected_dim1 = padded(128) * 4;
  if (tfrmd.dim1 != expected_dim1) {
    fprintf(stderr,
        "FAIL: LSTM normal dim1: expected %" PRIu32 " got %" PRIu32 "\n",
        expected_dim1, tfrmd.dim1);
    return 1;
  }
  printf("PASS: LSTM normal case dim1=%" PRIu32 "\n", tfrmd.dim1);
  return 0;
}

// ---------------------------------------------------------------------------
// Normal GRU: val=192, padded=192, dim1 = 192*3 = 576
// ---------------------------------------------------------------------------
static int testGRUNormalCase() {
  zdnn_tensor_desc pre, tfrmd;
  memset(&pre, 0, sizeof(pre));
  memset(&tfrmd, 0, sizeof(tfrmd));

  set_info_pre_transformed_desc(&pre, ZDNN_2DS, ZDNN_DLFLOAT16, {1, 192});

  zdnn_concat_info info = RNN_TYPE_GRU | PREV_LAYER_UNI | USAGE_BIASES;
  zdnn_status st = generate_transformed_desc_concatenated(&pre, info, &tfrmd);
  if (st != ZDNN_OK) {
    fprintf(stderr, "FAIL: GRU normal case: expected ZDNN_OK, got %d\n", st);
    return 1;
  }
  uint32_t expected_dim1 = padded(192) * 3;
  if (tfrmd.dim1 != expected_dim1) {
    fprintf(stderr,
        "FAIL: GRU normal dim1: expected %" PRIu32 " got %" PRIu32 "\n",
        expected_dim1, tfrmd.dim1);
    return 1;
  }
  printf("PASS: GRU normal case dim1=%" PRIu32 "\n", tfrmd.dim1);
  return 0;
}

// ---------------------------------------------------------------------------
// LSTM overflow: val chosen so PADDED(val)*4 > UINT32_MAX.
//
// PADDED(val) = ceil(val/64)*64.  We need ceil(val/64)*64 * 4 > 2^32.
// Smallest such val: ceil(val/64) >= 2^32/4/64 = 16777216 => val = 1073741824
// (= 2^30).  Before the fix this wrapped to 0 and the function returned 0,
// causing the concatenated descriptor to have dim1=0 and no downstream error.
// After the fix get_rnn_concatenated_dim1 returns 0, and
// generate_transformed_desc_concatenated propagates ZDNN_INVALID_SHAPE.
// ---------------------------------------------------------------------------
static int testLSTMOverflowRejected() {
  zdnn_tensor_desc pre, tfrmd;
  memset(&pre, 0, sizeof(pre));
  memset(&tfrmd, 0, sizeof(tfrmd));

  // val = 2^30; PADDED(2^30) = 2^30 (already 64-aligned);
  // PADDED * 4 = 2^32 > UINT32_MAX => overflow.
  const uint32_t overflow_val = 1073741824u; // 2^30
  set_info_pre_transformed_desc(
      &pre, ZDNN_2DS, ZDNN_DLFLOAT16, {1, (int64_t)overflow_val});

  zdnn_concat_info info = RNN_TYPE_LSTM | PREV_LAYER_UNI | USAGE_BIASES;
  zdnn_status st = generate_transformed_desc_concatenated(&pre, info, &tfrmd);
  if (st == ZDNN_OK) {
    fprintf(stderr,
        "FAIL: LSTM overflow case should return ZDNN_INVALID_SHAPE, "
        "got ZDNN_OK with dim1=%" PRIu32 " — f031 regression\n",
        tfrmd.dim1);
    return 1;
  }
  printf("PASS: LSTM overflow correctly rejected (status=%d)\n", (int)st);
  return 0;
}

// ---------------------------------------------------------------------------
// GRU overflow: PADDED(val)*3 > UINT32_MAX.
// Smallest such val: ceil(val/64)*64 > 2^32/3 => padded > 1431655765 =>
// ceil(val/64) >= 22369622 => val = 22369622*64 = 1431655808.
// ---------------------------------------------------------------------------
static int testGRUOverflowRejected() {
  zdnn_tensor_desc pre, tfrmd;
  memset(&pre, 0, sizeof(pre));
  memset(&tfrmd, 0, sizeof(tfrmd));

  const uint32_t overflow_val = 1431655808u; // ceil(x/64)*64 * 3 > 2^32
  set_info_pre_transformed_desc(
      &pre, ZDNN_2DS, ZDNN_DLFLOAT16, {1, (int64_t)overflow_val});

  zdnn_concat_info info = RNN_TYPE_GRU | PREV_LAYER_UNI | USAGE_BIASES;
  zdnn_status st = generate_transformed_desc_concatenated(&pre, info, &tfrmd);
  if (st == ZDNN_OK) {
    fprintf(stderr,
        "FAIL: GRU overflow case should return ZDNN_INVALID_SHAPE, "
        "got ZDNN_OK with dim1=%" PRIu32 " — f031 regression\n",
        tfrmd.dim1);
    return 1;
  }
  printf("PASS: GRU overflow correctly rejected (status=%d)\n", (int)st);
  return 0;
}

int main(int /*argc*/, char * /*argv*/[]) {
  int rc = 0;
  rc += testLSTMNormalCase();
  rc += testGRUNormalCase();
  rc += testLSTMOverflowRejected();
  rc += testGRUOverflowRejected();
  return rc;
}
