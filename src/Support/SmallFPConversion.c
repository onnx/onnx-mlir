#include "SmallFPConversion.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

// Defines variable TO of type TO_TYPE and copies bytes from variable FROM.
// Using memcpy because the simpler definition
//
//   #define BIT_CAST(TO_TYPE, TO, FROM) TO_TYPE TO = *(const TO_TYPE *)&FROM
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

// Implementation adapted from https://stackoverflow.com/a/3542975

float om_f16_to_f32(uint16_t u16) {
  static const int f32_sig_bits = 23;
  static const int f32_exp_bits = 8;
  static const int f32_bits = f32_sig_bits + f32_exp_bits + 1;
  static const int f32_exp_max = (1 << f32_exp_bits) - 1;
  static const int f32_exp_bias = f32_exp_max >> 1;
  static const uint32_t f32_inf = ((uint32_t)f32_exp_max) << f32_sig_bits;

  static const int f16_sig_bits = 10;
  static const int f16_exp_bits = 5;
  static const int f16_bits = f16_sig_bits + f16_exp_bits + 1;
  static const int f16_exp_max = (1 << f16_exp_bits) - 1;
  static const int f16_exp_bias = f16_exp_max >> 1;
  static const int f16_sign = ((uint16_t)1) << (f16_bits - 1);
  static const uint16_t f16_inf = ((uint16_t)f16_exp_max) << f16_sig_bits;

  static const int sig_diff = f32_sig_bits - f16_sig_bits;
  static const int bit_diff = f32_bits - f16_bits;
  static const uint32_t bias_mul = ((uint32_t)(2 * f32_exp_bias - f16_exp_bias))
                                   << f32_sig_bits;
  uint32_t bits = u16;
  uint32_t sign = bits & f16_sign; // save sign
  bits ^= sign;                    // clear sign
  bool is_norm = bits < f16_inf;
  bits = (sign << bit_diff) | (bits << sig_diff);
  BIT_CAST(float, bits_f32, bits);
  BIT_CAST(float, bias_mul_f32, bias_mul);
  float val_f32 = bits_f32 * bias_mul_f32;
  BIT_CAST(uint32_t, val, val_f32);
  val |= is_norm ? 0 : f32_inf;
  BIT_CAST(float, f32, val);
  return f32;
}

uint16_t om_f32_to_f16(float f32) {
  static const int f32_sig_bits = 23;
  static const int f32_exp_bits = 8;
  static const int f32_bits = f32_sig_bits + f32_exp_bits + 1;
  static const int f32_exp_max = (1 << f32_exp_bits) - 1;
  static const int f32_exp_bias = f32_exp_max >> 1;
  static const int f32_sign = ((uint32_t)1) << (f32_bits - 1);
  static const uint32_t f32_inf = ((uint32_t)f32_exp_max) << f32_sig_bits;

  static const int f16_sig_bits = 10;
  static const int f16_exp_bits = 5;
  static const int f16_bits = f16_sig_bits + f16_exp_bits + 1;
  static const int f16_exp_max = (1 << f16_exp_bits) - 1;
  static const int f16_exp_bias = f16_exp_max >> 1;
  static const uint16_t f16_inf = ((uint16_t)f16_exp_max) << f16_sig_bits;
  static const uint16_t f16_qnan = f16_inf | (f16_inf >> 1);

  static const int sig_diff = f32_sig_bits - f16_sig_bits;
  static const int bit_diff = f32_bits - f16_bits;
  static const uint32_t bias_mul = ((uint32_t)f16_exp_bias) << f32_sig_bits;
  BIT_CAST(float, bias_mul_f32, bias_mul);
  BIT_CAST(uint32_t, bits, f32);
  uint32_t sign = bits & f32_sign; // save sign
  bits ^= sign;                    // clear sign
  bool is_nan = f32_inf < bits;    // compare before rounding!!

  // round:
  {
    static const uint32_t min_norm =
        ((uint32_t)(f32_exp_bias - f16_exp_bias + 1)) << f32_sig_bits;
    static const uint32_t sub_rnd =
        f16_exp_bias < sig_diff
            ? 1u << (f32_sig_bits - 1 + f16_exp_bias - sig_diff)
            : ((uint32_t)(f16_exp_bias - sig_diff)) << f32_sig_bits;
    BIT_CAST(float, sub_rnd_f32, sub_rnd);
    static const uint32_t sub_mul = ((uint32_t)(f32_exp_bias + sig_diff))
                                    << f32_sig_bits;
    BIT_CAST(float, sub_mul_f32, sub_mul);
    bool is_sub = bits < min_norm;
    BIT_CAST(float, norm_f32, bits);
    float subn_f32 = norm_f32;
    subn_f32 *= sub_rnd_f32;  // round subnormals
    subn_f32 *= sub_mul_f32;  // correct subnormal exp
    norm_f32 *= bias_mul_f32; // fix exp bias
    BIT_CAST(uint32_t, norm, norm_f32);
    bits = norm;
    bits += (bits >> sig_diff) & 1;     // add tie breaking bias
    bits += (1u << (sig_diff - 1)) - 1; // round up to half
    BIT_CAST(uint32_t, subn, subn_f32);
    if (is_sub)
      bits = subn;
  }

  bits >>= sig_diff; // truncate
  if (f16_inf < bits)
    bits = f16_inf; // fix overflow
  if (is_nan)
    bits = f16_qnan;
  bits |= sign >> bit_diff; // restore sign
  return bits;
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
  u32 += 32767 + ((u32 & 0x1FFFF) == 0x18000) * 0x10000;
  uint16_t u16 = u32 >> 16;
  if ((u16 & 0x7FFF) == 0x7F80 && isnan(f32))
    return u16 + 0x40; // NAN
  if ((u16 & 0x7FFF) == 0 && isnan(f32))
    return u16 - 1; // NAN
  return u16;
}
