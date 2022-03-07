/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- convert.c ----------------------------------------------------===//
//
// Copyright 2020-2022 IBM
//
// =============================================================================
//
// This file contains conversions for floating point to ZDNN formats
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define UNSIGNED_NINF 0x7FFF

#define FLAG_INVALID 1 // Ignore and continue computation
#define FLAG_OVERFLOW                                                          \
  3 // Return to higher software stack. Data out of range for DLFLT16/AIU
#define FLAG_UNDERFLOW 4 // Ignore and continue computation

/// IEEE FP32 to DLFLT16 conversion with all flags
/// Produces output in dlfl16 format. Returns FP flags as return value
int zdnn_fl32_to_dlflt16(uint32_t *input, uint16_t *output) {
  uint32_t mantissa;
  int16_t exponent;
  int16_t sign;

  sign = (*input & 0x80000000) >>
         16; // Shift the sign in MSB bit of DLFLT16 output
  exponent = (*input & 0x7F800000) >> 23;
  mantissa =
      (*input & 0x007FFFFF); // Format: xxxx xxxx 1.uuu uuuu uuuu uuuu uuuu uuuu

  if (0)
    printf(
        "Got sign:%d, raw_exp=%d, raw_mantissa=%d\n", sign, exponent, mantissa);

  if (exponent ==
      0xff) { // NaN and INFINITY cases. Input was already out of range
    *output = 0x7fff | sign;
    if (mantissa)
      return FLAG_INVALID; // NaN
    else
      return FLAG_OVERFLOW; // INFINITY
  }

  if (exponent == 0x00) { // Denormal number cases
    *output = 0 | sign;
    if (mantissa)
      return FLAG_UNDERFLOW; // Underflow rounded to zero
    else
      return 0; // Exact zero
  }

  // Calculing exponent in DLFLT16 target format
  exponent = exponent - 127 + 31;

  if (exponent < 0) { // Exponent straight out of range in target format
    *output = 0 | sign;
    return FLAG_UNDERFLOW; // Underflow rounded to zero
  }

  // Rounding mantissa away from zero and detecting rounding-sourced exponent
  // overflow
  mantissa += 0x00800000; // Set implicit 1 before the coma
  mantissa += 0x00002000; // Add 1 at 10th position after the coma for rounding
  if (mantissa &
      0x01000000) { // If carry rippled all the way to before the coma
    exponent++;     // Add 1 to exponent
    mantissa >>= 1; // Shift mantissa by one to right [and truncate]
  }

  if ((exponent == 0)) {
    if ((mantissa & 0x007fe000) ==
        0x00004000) { // Rounding pushed the value to the smallest representable
                      // number
      *output =
          0x0001 | sign; // Result is already correct, need this for flag only
      return FLAG_UNDERFLOW;
    }
    if ((mantissa & 0x007fc000) == 0) { // Still zero after rounding
      *output = 0 | sign; // Result is already correct, need this for flag only
      return FLAG_UNDERFLOW;
    }
  }

  if ((exponent == 63) &&
      ((mantissa & 0x007fc000) ==
          0x007fc000)) { // We have hit exactly the NINF encoding after rounding
    *output =
        0x7fff |
        sign; // Result is already accurate but need to set the OVERFLOW flag
    return FLAG_OVERFLOW;
  }

  if (exponent >= 64) { // Exponent out of range after rounding
    *output = 0x7fff | sign;
    return FLAG_OVERFLOW;
  }

  *output = sign | (exponent << 9) | ((mantissa & 0x007fc000) >> 14);
  return 0;
}

/// IEEE FP32 to DLFLT16 conversion without flags
/// Produces output in dlfl16 format. Returns 1 if SW should stop computation
/// (out of range)
int zdnn_fl32_to_dlflt16_noflag(uint32_t *input, uint16_t *output) {
  uint32_t mantissa;
  int16_t exponent;
  int16_t sign;

  sign = (*input & 0x80000000) >>
         16; // Shift the sign in MSB bit of DLFLT16 output
  exponent = (*input & 0x7F800000) >> 23;
  mantissa =
      (*input & 0x007FFFFF); // Format: xxxx xxxx 1.uuu uuuu uuuu uuuu uuuu uuuu

  if (0)
    printf(
        "Got sign:%d, raw_exp=%d, raw_mantissa=%d\n", sign, exponent, mantissa);

  if (exponent ==
      0xff) { // NaN and INFINITY cases. Input was already out of range
    *output = 0x7fff | sign;
    return 1;
  }

  // Calculing exponent in DLFLT16 target format
  exponent = exponent - 127 + 31;

  if (exponent < 0) { // Exponent straight out of range in target format
    *output = 0 | sign;
    return 0;
  }

  // Rounding mantissa away from zero and detecting rounding-sourced exponent
  // overflow
  mantissa += 0x00800000; // Set implicit 1 before the coma
  mantissa += 0x00002000; // Add 1 at 10th position after the coma for rounding
  if (mantissa &
      0x01000000) { // If carry rippled all the way to before the coma
    exponent++;     // Add 1 to exponent
    mantissa >>= 1; // Shift mantissa by one to right [and truncate]
  }

  // This test could be done after the final output calculation below by just
  // testing output value
  if ((exponent == 63) &&
      ((mantissa & 0x007fc000) ==
          0x007fc000)) { // We have hit exactly the NINF encoding after rounding
    *output = 0x7fff | sign; // So we are out of range
    return 1;
  }

  if (exponent >= 64) { // Exponent out of range after rounding
    *output = 0x7fff | sign;
    return 1;
  }

  *output = sign | (exponent << 9) | ((mantissa & 0x007fc000) >> 14);
  return 0;
}
