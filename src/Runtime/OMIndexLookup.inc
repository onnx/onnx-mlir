/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- OMIndexLookup.inc - OMIndexLookup C/C++ Implementation -------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ implementation of the OMIndexLookup functions.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Perform a 32-bit FNV (Fowler-Noll-Vo) hash on the given string.
static inline uint32_t hash_string(uint32_t hval, const char *str) {
  uint32_t prime = 0x01000193;
  hval = (hval == 0) ? prime : hval;

  int32_t len = strlen(str);
  for (int32_t i = 0; i < len; ++i) {
    char c = str[i];
    hval *= prime;
    hval ^= c;
  }
  return hval;
}

// Adaptation of 32-bit FNV for int64_t values.
static inline uint32_t hash_int64(uint32_t hval, int64_t val) {
  char str[20];
  int num_chars_written = snprintf(str, sizeof(str), "%lld", (long long)val);
  assert(num_chars_written >= 0 && "snprintf write error to str");
  return hash_string(hval, str);
}

/// Return the index (i.e. value) of the given string \p str in a perfect hash
/// table described by the arrays \p G and \p V. The arrays length is given by
/// \p dictSize. The index returned is valid if the string \p str provided is
/// garanteed to be in the dictionary described by \p G and \p V.
#ifdef __cplusplus
extern "C"
#endif
    uint64_t
    find_index_str(const char *str, const int32_t G[], const int32_t V[],
        int32_t dictSize) {
  assert(str && G && V && dictSize > 0);
  int32_t d = G[hash_string(0, str) % dictSize];
  int64_t index = (d < 0) ? V[-d - 1] : V[hash_string(d, str) % dictSize];
  assert(index >= 0 && index < dictSize);
  return index;
}

/// Return the index (i.e. value) of the given integer \p val in a perfect hash
/// table described by the arrays \p G and \p V. The arrays length is given by
/// \p dictSize. The index returned is valid if the value provided \p val is
/// garanteed to be in the 'dictionary' described by \p G and \p V.
#ifdef __cplusplus
extern "C"
#endif
    uint64_t
    find_index_i64(
        int64_t val, const int32_t G[], const int32_t V[], int32_t dictSize) {
  assert(G && V && dictSize > 0);
  int32_t d = G[hash_int64(0, val) % dictSize];
  int64_t index = (d < 0) ? V[-d - 1] : V[hash_int64(d, val) % dictSize];
  assert(index >= 0 && index < dictSize);
  return index;
}
