/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- TestStickifyRNN.cpp - Unit tests for f031 RNN dim1 overflow -------===//
//
// Copyright 2026 The IBM Research Authors.
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
// indirectly through generate_transformed_desc_concatenated(), which now
// propagates ZDNN_INVALID_SHAPE when get_rnn_concatenated_dim1() returns 0.
// Test values are chosen so that PADDED(val)*gates truncates to a non-zero
// uint32 in the old code (ZDNN_OK, no error) but overflows in the new code
// (ZDNN_INVALID_SHAPE returned), making the tests a true regression sentinel.
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
#include "zdnn.h"
}

#include "Accelerators/NNPA/Support/Stickify/Stickify.hpp"

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
// PADDED(val) = ceil(val/64)*64.  We need PADDED(val)*4 to both exceed 2^32
// AND truncate to a non-zero uint32 so the test distinguishes old (no-fix)
// from new (fixed) behavior.  val=2^30 gives PADDED*4=2^32 which truncates
// to 0 even without the fix — unhelpful.
//
// val = 2^30 + 64 = 1073741888 gives PADDED*4 = 4294967552.
//   Old code (uint32 multiply): truncates to 256 — non-zero, ZDNN_OK returned.
//   New code (uint64 guard):    detects overflow, returns 0 sentinel, and
//   generate_transformed_desc_concatenated returns ZDNN_INVALID_SHAPE.
// ---------------------------------------------------------------------------
static int testLSTMOverflowRejected() {
  zdnn_tensor_desc pre, tfrmd;
  memset(&pre, 0, sizeof(pre));
  memset(&tfrmd, 0, sizeof(tfrmd));

  // PADDED(1073741888)*4 = 4294967552 > UINT32_MAX; truncates to 256 without
  // fix (non-zero, would silently succeed), returns ZDNN_INVALID_SHAPE with fix.
  const uint32_t overflow_val = 1073741888u; // 2^30 + 64
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
//
// val = 1431655808 = 22369622*64.  PADDED*3 = 4294967424 > UINT32_MAX.
//   Old code (uint32 multiply): truncates to 128 — non-zero, ZDNN_OK returned.
//   New code (uint64 guard):    detects overflow, returns 0 sentinel, and
//   generate_transformed_desc_concatenated returns ZDNN_INVALID_SHAPE.
// ---------------------------------------------------------------------------
static int testGRUOverflowRejected() {
  zdnn_tensor_desc pre, tfrmd;
  memset(&pre, 0, sizeof(pre));
  memset(&tfrmd, 0, sizeof(tfrmd));

  // PADDED(1431655808)*3 = 4294967424 > UINT32_MAX; truncates to 128 without
  // fix (non-zero, would silently succeed), returns ZDNN_INVALID_SHAPE with fix.
  const uint32_t overflow_val = 1431655808u; // 22369622 * 64
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
