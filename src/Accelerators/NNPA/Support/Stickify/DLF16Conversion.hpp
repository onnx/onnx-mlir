/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- DLF16Conversion.hpp - DLF16 Conversion -----------------------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// NNP-data-format 1 conversions
// Originally written by Joachim_von_Buttlar@de.ibm.com
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DLF16_H
#define ONNX_MLIR_DLF16_H
#include <arpa/inet.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#define BFLAST(mask) ((mask) & (1 + ~(mask)))
#define BFGET(w, mask) (((w) & (mask)) / BFLAST(mask))
#define BFPUT(w, mask, value)                                                  \
  ((w) = ((w) & ~(mask)) | (((value)*BFLAST(mask)) & (mask)))

static constexpr uint8_t VIC_INVALID = 1;
static constexpr uint8_t VIC_DIVIDE = 2;
static constexpr uint8_t VIC_OVERFLOW = 3;
static constexpr uint8_t VIC_UNDERFLOW = 4;
static constexpr uint8_t VIC_INEXACT = 5;

template <typename UIntType, unsigned ExponentBits, unsigned FractionBits>
class FPFormat {

  static_assert(sizeof(UIntType) * 8 == 1 + ExponentBits + FractionBits,
      "Inconsistent FPFormat declaration");

public:
  static constexpr UIntType One = 1;
  static constexpr unsigned EXPONENT_BITS = ExponentBits;
  static constexpr unsigned FRACTION_BITS = FractionBits;
  static constexpr UIntType SIGN = One << ExponentBits << FractionBits;
  static constexpr UIntType EXPONENT = ((One << ExponentBits) - 1)
                                       << FractionBits;
  static constexpr signed EXPONENT_BIAS = (One << (ExponentBits - 1)) - 1;
  static constexpr UIntType FRACTION = (One << FractionBits) - 1;
};

class NNP1 : public FPFormat<uint16_t, 6, 9> {

private:
  uint16_t item;

public:
  static constexpr uint16_t NINF = NNP1::EXPONENT | NNP1::FRACTION;
  static constexpr uint16_t NMAX = NNP1::EXPONENT | (NNP1::FRACTION - 1);

  NNP1(const int &binary = 0) : item(htons(binary)) {}

  uint16_t uint() const { return ntohs(this->item); }

  void convert(const float &fp, unsigned *vic = nullptr);

  float convert(unsigned *vic = nullptr) const;

  NNP1(const float &fp) { this->convert(fp); } // NNP1

  NNP1(const double &fp) { this->convert(float(fp)); } // NNP1

  operator float() const { return this->convert(); } // operator float

  operator double() const { return double(this->convert()); } // operator double

  bool is_zero() const { return (this->uint() & ~NNP1::SIGN) == 0; } // is_zero

  bool is_positive() const {
    return (this->uint() & NNP1::SIGN) == 0;
  } // is_positive

  bool is_ninf() const {
    return (this->uint() & ~NNP1::SIGN) == NNP1::NINF;
  } // is_ninf

}; // class NNP1

class FP32 : public FPFormat<uint32_t, 8, 23> {
public:
  static constexpr uint32_t NNP1_ROUND =
      1 << (FP32::FRACTION_BITS - NNP1::FRACTION_BITS - 1);
  static constexpr uint32_t NNP1_NMAX =
      (((1 << NNP1::EXPONENT_BITS) - 1 + FP32::EXPONENT_BIAS -
           NNP1::EXPONENT_BIAS)
          << FP32::FRACTION_BITS) |
      (((1 << NNP1::FRACTION_BITS) - 2)
          << (FP32::FRACTION_BITS - NNP1::FRACTION_BITS)) |
      (FP32::NNP1_ROUND - 1);
};

inline float NNP1::convert(unsigned *vic) const {
  if (vic)
    *vic = 0;

  float fp;

  if (this->is_zero()) {

    fp = this->is_positive() ? +0.0f : -0.0f;
  } else if (this->is_ninf()) {

    fp = NAN;

    if (vic)
      *vic = VIC_INVALID;
  } else {

    uint32_t fp32 = FP32::SIGN * BFGET(this->uint(), NNP1::SIGN);

    BFPUT(fp32, FP32::EXPONENT,
        BFGET(this->uint(), NNP1::EXPONENT) - NNP1::EXPONENT_BIAS +
            FP32::EXPONENT_BIAS);
    BFPUT(fp32, FP32::FRACTION,
        BFGET(this->uint(), NNP1::FRACTION)
            << (FP32::FRACTION_BITS - NNP1::FRACTION_BITS));
    memcpy(&fp, &fp32, sizeof(fp));
  }

  return fp;
} // NNP1::convert

inline void NNP1::convert(const float &fp, unsigned *vic) {
  if (vic)
    *vic = 0;

  uint32_t fp32;
  memcpy(&fp32, &fp, sizeof(fp32));

  signed nnp1_biased_exponent =
      BFGET(fp32, FP32::EXPONENT) - FP32::EXPONENT_BIAS + NNP1::EXPONENT_BIAS;
  uint32_t fraction = BFGET(fp32, FP32::FRACTION) + FP32::NNP1_ROUND;

  if (fraction > FP32::FRACTION) {
    fraction = 0;
    nnp1_biased_exponent++;
  }

  uint16_t uint = NNP1::SIGN * BFGET(fp32, FP32::SIGN);

  if (nnp1_biased_exponent >= 0) {
    if ((fp32 & ~FP32::SIGN) <= FP32::NNP1_NMAX) {
      BFPUT(uint, NNP1::EXPONENT, nnp1_biased_exponent);
      BFPUT(uint, NNP1::FRACTION,
          fraction >> (FP32::FRACTION_BITS - NNP1::FRACTION_BITS));
    } else {
      uint |= NNP1::NINF;
      if (vic) {
        if ((fp32 & FP32::EXPONENT) == FP32::EXPONENT) {
          if (fp32 & FP32::FRACTION) {
            *vic = VIC_INVALID;
          } else {
            *vic = VIC_OVERFLOW;
          }
        }
      }
    }
  } else {
    if (vic) {
      if (fp32 & ~FP32::SIGN) {
        *vic = VIC_UNDERFLOW;
      }
    }
  }

  *this = uint;
} // NNP1::convert
#endif
