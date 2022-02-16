/***PROPRIETARY_STATEMENT********************************************
 *
 * IBM CONFIDENTIAL
 *
 * OCO SOURCE MATERIALS
 *
 * 5650-ZOS
 *
 * COPYRIGHT IBM CORP. 2020
 *
 * THE SOURCE CODE FOR THIS PROGRAM IS NOT PUBLISHED OR OTHERWISE
 * DIVESTED OF ITS TRADE SECRETS, IRRESPECTIVE OF WHAT HAS BEEN
 * DEPOSITED WITH THE U.S. COPYRIGHT OFFICE.
 *
 * STATUS = HBB77D0
 *
 ********************************************************************
 *
 */

#include "src/Transform/ZHigh/Stickify/convert_sw.h"
#include "src/Transform/ZHigh/Stickify/convert.h"
#include <stdint.h>
#include <stdio.h>

/*
 * functions for creating fp16/bfloat/dlf16 raw data via C float,
 * primarily for tests
 */

// fp16 <-> float functions ------------------------------------------------

uint16_t float_to_fp16(float f) { return float16_t(f).uint(); }
float fp16_to_float(uint16_t u) { return float16_t(u); }

// fp32 <-> float functions ------------------------------------------------

float float_to_fp32(float f) { return f; }
float fp32_to_float(float f) { return f; }

// bfloat <-> float functions ------------------------------------------------

// used to get around "breaking strict-aliasing rules" so that we can
// manipulate the bits within a float
typedef union uint32_float_u {
  float f;
  uint32_t u;
} uint32_float_u;

// we simply chop off the last 16 mantissa-bits
uint16_t float_to_bfloat(float f) { return (*(uint32_float_u *)&f).u >> 16; }

// we simply add 16 0's as mantissa-bits
float bfloat_to_float(uint16_t f) {
  uint32_t u = f << 16;
  return (*(uint32_float_u *)&u).f;
}

// dlf16 <-> float functions  --------------------------------------------

uint16_t float_to_dlf16(float f) { return dlfloat16_t(f).uint(); }
float dlf16_to_float(uint16_t d) { return dlfloat16_t(d).convert(); }

/*
 * C wrappers for dlfloat-converter.h functions, convert various things
 * to/from dlf16
 */

uint64_t fp32_to_dlf16_in_stride(float *fp32_data, uint16_t *dflt16_data,
    uint64_t num_fields, uint32_t input_stride) {

  for (uint64_t i = 0; i < num_fields; i++) {
    dflt16_data[i] = dlfloat16_t(fp32_data[i * input_stride]).uint();
  }

  return num_fields;
}

uint64_t dlf16_to_fp32_in_stride(uint16_t *dflt16_data, float *fp32_data,
    uint64_t num_fields, uint32_t input_stride) {

  for (uint64_t i = 0; i < num_fields; i++) {
    fp32_data[i] = dlfloat16_t(dflt16_data[i * input_stride]).convert();
  }

  return num_fields;
}

uint64_t fp16_to_dlf16_in_stride(uint16_t *fp16_data, uint16_t *dflt16_data,
    uint64_t num_fields, uint32_t input_stride)

{

  for (uint64_t i = 0; i < num_fields; i++) {
    dflt16_data[i] = float16_t(fp16_data[i * input_stride]).convert().uint();
  }

  return num_fields;
}

uint64_t dlf16_to_fp16_in_stride(uint16_t *dflt16_data, uint16_t *fp16_data,
    uint64_t num_fields, uint32_t input_stride)

{

  for (uint64_t i = 0; i < num_fields; i++) {
    fp16_data[i] = float16_t(dlfloat16_t(dflt16_data[i * input_stride])).uint();
  }

  return num_fields;
}

uint64_t bfloat_to_dlf16_in_stride(uint16_t *bflt_data, uint16_t *dflt16_data,
    uint64_t num_fields, uint32_t input_stride) {

  for (uint64_t i = 0; i < num_fields; i++) {
    dflt16_data[i] =
        dlfloat16_t(bfloat_to_float(bflt_data[i * input_stride])).uint();
  }

  return num_fields;
}

uint64_t dlf16_to_bfloat_in_stride(uint16_t *dflt16_data, uint16_t *bflt_data,
    uint64_t num_fields, uint32_t input_stride) {

  for (uint64_t i = 0; i < num_fields; i++) {
    bflt_data[i] =
        float_to_bfloat(dlfloat16_t(dflt16_data[i * input_stride]).convert());
  }

  return num_fields;
}

uint64_t fp32_to_dlf16(
    float *fp32_data, uint16_t *dflt16_data, uint64_t num_fields) {
  return fp32_to_dlf16_in_stride(fp32_data, dflt16_data, num_fields, 1);
}

uint64_t dlf16_to_fp32(
    uint16_t *dflt16_data, float *fp32_data, uint64_t num_fields) {
  return dlf16_to_fp32_in_stride(dflt16_data, fp32_data, num_fields, 1);
}

uint64_t fp16_to_dlf16(
    uint16_t *fp16_data, uint16_t *dflt16_data, uint64_t num_fields) {
  return fp16_to_dlf16_in_stride(fp16_data, dflt16_data, num_fields, 1);
}

uint64_t dlf16_to_fp16(
    uint16_t *dflt16_data, uint16_t *fp16_data, uint64_t num_fields) {
  return dlf16_to_fp16_in_stride(dflt16_data, fp16_data, num_fields, 1);
}

uint64_t bfloat_to_dlf16(
    uint16_t *bflt_data, uint16_t *dflt16_data, uint64_t num_fields) {
  return bfloat_to_dlf16_in_stride(bflt_data, dflt16_data, num_fields, 1);
}

uint64_t dlf16_to_bfloat(
    uint16_t *dflt16_data, uint16_t *bflt_data, uint64_t num_fields) {
  return dlf16_to_bfloat_in_stride(dflt16_data, bflt_data, num_fields, 1);
}
