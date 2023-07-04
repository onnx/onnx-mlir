#include "SmallFPConversion.h"

#include <assert.h>
#include <string.h>

// Defines variable TO of type TO_TYPE and copies bytes from variable FROM.
// Using memcpy because the simpler definition
//
//   #define BIT_CAST(TYPE, TO, FROM) TYPE TO = *(const TYPE *)FROM
//
// might violate the rules about strict aliasing in C++.
#define BIT_CAST(TO_TYPE, TO, FROM)                                            \
  TO_TYPE TO;                                                                  \
  static_assert(sizeof(TO) == sizeof(FROM), "only bit cast same sizes");       \
  memcpy(&TO, &FROM, sizeof(FROM))

// When the CPU is known to support native conversion between float and float_16
// we define FLOAT16_TO_FLOAT32(u16) and FLOAT32_TO_FLOAT16(f32) macros, used in
// class float_16 below to override the slow default APFloat-based conversions.
//
// FLOAT16_TO_FLOAT32(u16) takes a bit cast float_16 number as uint16_t and
// evaluates to a float.
//
// FLOAT32_TO_FLOAT16(f32) takes a float f32 number and evaluates to a bit cast
// float_16 number as uint16_t.
//
#if defined(__x86_64__) && defined(__F16C__)
// On x86-64 build config -DCMAKE_CXX_FLAGS=-march=native defines __F16C__.

// https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-9/details-about-intrinsics-for-half-floats.html
#include <immintrin.h>

float om_f16_to_f32(uint16_t u16) { return _cvtsh_ss(u16); }

uint16_t om_f32_to_f16(float f32) {
  return _cvtss_sh(f32, /*ROUND TO NEAREST EVEN*/ 0);
}

#elif defined(__ARM_FP16_FORMAT_IEEE)
// On MacBook Pro no build config is needed to define __ARM_FP16_FORMAT_IEEE.

// https://arm-software.github.io/acle/main/acle.html#half-precision-floating-point

float om_f16_to_f32(uint16_t u16) {
  BIT_CAST(__fp16, f16, u16);
  return (float)f16;
}

uint16_t om_f32_to_f16(float f32) {
  __fp16 f16 = (__fp16)f32;
  BIT_CAST(uint16_t, u16, f16);
  return u16;
}

#else

// Implementation adapted from https://stackoverflow.com/a/60047308

float om_f16_to_f32(uint16_t u16) {
  uint32_t e = (u16 & 0x7C00) >> 10; // exponent
  uint32_t m = (u16 & 0x03FF) << 13; // mantissa
  // evil log2 bit hack to count leading zeros in denormalized format:
  float m_float = (float)m;
  BIT_CAST(uint32_t, m_float_bits, m_float);
  uint32_t v = m_float_bits >> 23;
  uint32_t u32 = // sign : normalized : denormalized
      (u16 & 0x8000u) << 16 | (e != 0) * ((e + 112) << 23 | m) |
      ((e == 0) & (m != 0)) *
          ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000));
  BIT_CAST(float, f32, u32);
  return f32;
}

uint16_t om_f32_to_f16(float f32) {
  // round-to-nearest-even: add last bit after truncated mantissa
  BIT_CAST(uint32_t, u32, f32);
  uint32_t b = u32 + 0x00001000;
  uint32_t e = (b & 0x7F800000) >> 23; // exponent
  uint32_t m = b & 0x007FFFFF;         // mantissa
  // in line below: 0x007FF000 = 0x00800000 - 0x00001000
  //                           = decimal indicator flag - initial rounding
  return // sign : normalized : denormalized : saturate
      (b & 0x80000000u) >> 16 |
      (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
      ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
      (e > 143) * 0x7FFF;
}

#endif

// Implementation adapted from the answers to
// https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c

float om_bf16_to_f32(uint16_t u16) {
  uint32_t u32 = ((uint32_t)u16) << 16;
  BIT_CAST(float, f32, u32);
  return f32;
}

uint16_t om_f32_to_bf16(float f32) {
  BIT_CAST(uint32_t, u32, f32);
  return u32 >> 16;
}
