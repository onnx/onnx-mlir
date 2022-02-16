// zDNN ADDED BEGIN -----------------
// clang-format off

/*-------------------------------------------------------------------*/
/*                         IBM CONFIDENTIAL                          */
/*-------------------------------------------------------------------*/
/*                REMOVE THIS FILE BEFORE DISTRIBUTION!              */
/*-------------------------------------------------------------------*/

#ifdef ZOS
#define _TR1_C99                  // needed for NAN/INFINITY
#define _XOPEN_SOURCE_EXTENDED 1  // needed for htons() etc
#endif

// #ifndef MCEAIUFP_H
// #define MCEAIUFP_H
#ifndef CONVERT_SW_H
#define CONVERT_SW_H
// zDNN ADDED END -----------------
#include <algorithm>    // std::min, max
#include <arpa/inet.h>  // for endianness conversion
#include <cmath>        // for NAN
#include <stdint.h>
#include <string.h>
#include <stdio.h>

/*-------------------------------------------------------------------*/
/*                         IBM CONFIDENTIAL                          */
/*-------------------------------------------------------------------*/
/*  Project: System z Millicode Emulator                             */
/*  Author:  J.v.Buttlar                                             */
/*  Title:   AIU floating-point support                              */
/*-------------------------------------------------------------------*/
/*  Description:                                                     */
/*    Classes for floating-point formats used with AIU operations,   */
/*    conversion functions, and some mathematical operations         */
/*-------------------------------------------------------------------*/

// Bit fumbling macros
#define BFLAST(mask)                    \
    ((mask) & (1 + ~(mask)))
#define BFGET(w, mask)                  \
    ( ((w) & (mask)) / BFLAST(mask) )
#define BFPUT(w, mask, value)           \
    ( (w) = ((w) & ~(mask)) | (((value) * BFLAST(mask)) & (mask)) )

/* Vector interrupt codes in VIC field of VXC */

static const uint8_t  VIC_INVALID                   = 1;
static const uint8_t  VIC_DIVIDE                    = 2;
static const uint8_t  VIC_OVERFLOW                  = 3;
static const uint8_t  VIC_UNDERFLOW                 = 4;
static const uint8_t  VIC_INEXACT                   = 5;

// Lookup tables for mathematical function result

extern uint16_t dlfloat_lookup_log[65536];
extern uint16_t dlfloat_lookup_exp[65536];
extern uint16_t dlfloat_lookup_tanh[65536];
extern uint16_t dlfloat_lookup_sigmoid[65536];
extern uint16_t dlfloat_lookup_reciprocal[65536];

#if __cplusplus < 201103
#define constexpr const
#define nullptr NULL
#endif

// zDNN ADDED BEGIN -----------------
// simulate gcc's __builtin_clz, google'd code
#ifdef ZOS
int __builtin_clz(int x)
{
    unsigned y;
    int n = 32;
    y = x >> 16;
    if (y != 0) { n = n - 16; x = y; }
    y = x >> 8;
    if (y != 0) { n = n - 8; x = y; }
    y = x >> 4;
    if (y != 0) { n = n - 4; x = y; }
    y = x >> 2;
    if (y != 0) { n = n - 2; x = y; }
    y = x >> 1;
    if (y != 0)
        return n - 2;
    return n - x;
}
#endif
// zDNN ADDED END -----------------

// AIU floating-point formats

template <typename UIntType, unsigned ExponentBits, unsigned FractionBits>
class FPFormat {

#if __cplusplus >= 201103
    static_assert(sizeof(UIntType) * 8 == 1 + ExponentBits + FractionBits, "Inconsistent FPFormat declaration");
#endif

private:
    static constexpr UIntType   One = 1;

public:
    static constexpr unsigned   EXPONENT_BITS   = ExponentBits;
    static constexpr unsigned   FRACTION_BITS   = FractionBits;
    static constexpr UIntType   SIGN            =   One << ExponentBits << FractionBits;
    static constexpr UIntType   EXPONENT        = ((One << ExponentBits) - 1) << FractionBits;
    static constexpr signed     EXPONENT_BIAS   =  (One << (ExponentBits - 1)) - 1;
    static constexpr UIntType   FRACTION        =  (One << FractionBits) - 1;
};

class FP32: public  FPFormat<uint32_t, 8, 23> { };  // IEEE FP32

class DLF:  public  FPFormat<uint16_t, 6, 9> {
public:
    static constexpr uint16_t   NINF        = DLF::EXPONENT | DLF::FRACTION;
    static constexpr uint32_t   FP32_ROUND  = 1 << (FP32::FRACTION_BITS - DLF::FRACTION_BITS - 1);
};

class FP16: public  FPFormat<uint16_t, 5, 10> {     // IEEE FP16
public:
    static constexpr uint16_t   INF         = FP16::EXPONENT;
    static constexpr uint16_t   QNAN        = FP16::EXPONENT | ((FP16::FRACTION >> 1) + 1);
    static constexpr uint16_t   DLF_ROUND   = 1 << (FP16::FRACTION_BITS - DLF::FRACTION_BITS - 1);
    // zDNN ADDED BEGIN -----------------
    static constexpr uint16_t   NINF        = FP16::EXPONENT | FP16::FRACTION;
    static constexpr uint32_t   FP32_ROUND  = 1 << (FP32::FRACTION_BITS - FP16::FRACTION_BITS - 1);
    // zDNN ADDED END -----------------
};

class dlfloat16_t {

private:
    uint16_t    item;           // internal representation (big-endian)

public:
    // Construct big-endian dlfloat16_t from integer

    dlfloat16_t(const int binary = 0): item(htons(binary)) {}

    // Retrieve integer value, regardless of endianness

    uint16_t uint() const {
        return ntohs(this->item);
    }

    // Convert float to dlfloat16_t

    void convert(const float &fp,
                 unsigned    *vic = nullptr) {
        if (vic)
            *vic = 0;

        uint32_t    fp32;
        memcpy(&fp32, &fp, sizeof(fp32));

        signed      dlf_biased_exponent = BFGET(fp32, FP32::EXPONENT) - FP32::EXPONENT_BIAS + DLF::EXPONENT_BIAS;

        // Round to nearest with ties away from 0

        uint32_t    fraction = BFGET(fp32, FP32::FRACTION) + DLF::FP32_ROUND;

        if (fraction > FP32::FRACTION) {

            // Rounding caused fraction overflow:
            //                      +------------------- implied 1
            //                      v
            // original fraction    1.111 1111 111a bcde fghi jklm
            // + round              1.000 0000 0010 0000 0000 0000
            // overflows           10.000 0000 000a bcde fghi jklm
            // >> 1                 1.000 0000 0000 abcd efgh ijkl
            // (and adjust exponent)
            // fraction bits to use  .000 0000 00

            fraction = 0;
            dlf_biased_exponent++;
        }

        // Initialize result with sign

        uint16_t    uint = DLF::SIGN * BFGET(fp32, FP32::SIGN);

        if (dlf_biased_exponent >= 0) {

            if (dlf_biased_exponent < (1 << DLF::EXPONENT_BITS)) {

                // Insert exponent, adjusted to bias for DLFloat

                BFPUT(uint, DLF::EXPONENT, dlf_biased_exponent);

                // Insert leftmost part of BFP fraction

                BFPUT(uint, DLF::FRACTION, fraction >> (FP32::FRACTION_BITS - DLF::FRACTION_BITS));
            } else {

                // Too large for DLFloat; this also covers BFP infinity and NaN

                uint |= DLF::NINF;

                if (vic) {
                    if ((fp32 & FP32::EXPONENT) == FP32::EXPONENT) {
                        if (fp32 & FP32::FRACTION) {

                            // BFP NaN

                            *vic = VIC_INVALID;
                        } else {

                            // BFP infinity

                            *vic = VIC_OVERFLOW;
                        }
                    }
                }
            }
        } else {

            // Too small for DLFloat, leave exponent and fraction zero but keep sign

            if (vic)
                *vic = VIC_UNDERFLOW;
        }

        *this = uint;
    } // dlfloat16_t

    // Construct dlfloat16_t from float

    dlfloat16_t(const float &fp) {
        this->convert(fp);
    } // dlfloat16_t

    // Construct dlfloat16_t from double (e.g. constants)

    dlfloat16_t(const double &fp) {
        this->convert(float(fp));
    } // dlfloat16_t

    // Convert dlfloat16_t to float

    float convert(unsigned *vic = nullptr) const {

        if (vic)
            *vic = 0;

        float       fp;

        if (this->is_zero()) {

            // Propagate sign even for 0

            fp = this->is_positive() ? +0.0f : -0.0f;
        } else if (this->is_ninf()) {

            // Sign is not propagated from NINF to NaN

            fp = NAN;

            if (vic)
                *vic = VIC_INVALID;
        } else {

            // Initialize result with sign

            uint32_t    fp32 = FP32::SIGN * BFGET(this->uint(), DLF::SIGN);

            // Insert exponent, adjusted to bias for BFP short

            BFPUT(fp32, FP32::EXPONENT, BFGET(this->uint(), DLF::EXPONENT) - DLF::EXPONENT_BIAS + FP32::EXPONENT_BIAS);

            // Insert fraction, left adjusted

            BFPUT(fp32, FP32::FRACTION, BFGET(this->uint(), DLF::FRACTION) << (FP32::FRACTION_BITS - DLF::FRACTION_BITS));

            memcpy(&fp, &fp32, sizeof(fp));
        }

        return fp;
    } // convert

    operator float() const {
        return this->convert();
    } // operator float

    operator double() const {
        return double(this->convert());
    } // operator double

    // Test for +/- 0

    bool is_zero() const {
        return (this->uint() & ~DLF::SIGN) == 0;
    } // is_zero

    // Test sign bit

    bool is_positive() const {
        return (this->uint() & DLF::SIGN) == 0;
    } // is_positive

    // Test for +/- NINF

    bool is_ninf() const {
        return (this->uint() & ~DLF::SIGN) == DLF::NINF;
    } // is_ninf

    // Natural logarithm

    dlfloat16_t log() {
        return dlfloat_lookup_log[this->uint()];
    } // log

    // Exponential function

    dlfloat16_t exp() {
        return dlfloat_lookup_exp[this->uint()];
    } // exp

    // Rectified linear unit

    dlfloat16_t relu() {
        return this->is_ninf() ? *this : this->is_positive() ? *this : static_cast<dlfloat16_t>(0);
    } // relu

    // Hyperbolic tangent

    dlfloat16_t tanh() {
        return dlfloat_lookup_tanh[this->uint()];
    } // tanh

    // Sigmoid

    dlfloat16_t sigmoid() {
        return dlfloat_lookup_sigmoid[this->uint()];
    } // sigmoid

}; // class dlfloat16_t

static inline dlfloat16_t
operator - (const dlfloat16_t &a) {     // sign inversion

    return a.uint() ^ DLF::SIGN;
} // operator -

static inline dlfloat16_t
operator + (const dlfloat16_t &a, const dlfloat16_t &b) {

    dlfloat16_t result;

    if (b.is_ninf())
        result = b;
    else if (a.is_ninf())
        result = a;
    else
        result = float(a) + float(b);

    return result;
} // operator +

static inline dlfloat16_t
operator - (const dlfloat16_t &a, const dlfloat16_t &b) {

    dlfloat16_t result;

    if (a.is_ninf())
        result = a;
    else if (b.is_ninf())
        result = -b;
    else
        result = float(a) - float(b);

    return result;
} // operator -

static inline dlfloat16_t
operator * (const dlfloat16_t &a, const dlfloat16_t &b) {

    dlfloat16_t result;

    if (a.is_ninf() || b.is_ninf())
        result = DLF::NINF | ((a.uint() ^ b.uint()) & DLF::SIGN);
    else
        result = float(a) * float(b);

    return result;
} // operator *

static inline dlfloat16_t
operator / (const dlfloat16_t &a, const dlfloat16_t &b) {

    dlfloat16_t result;

    if (b.is_zero())
        result = DLF::NINF | ((a.uint() ^ b.uint()) & DLF::SIGN);
    else
        result = a * static_cast<dlfloat16_t>(dlfloat_lookup_reciprocal[b.uint()]);

    return result;
} // operator /

static inline dlfloat16_t
min(const dlfloat16_t &a, const dlfloat16_t &b) {

    dlfloat16_t result;

    if (a.is_ninf()) {
        if (!a.is_positive()) {
            result = a;
        } else if (b.is_ninf()) {
            result = b;
        } else {
            result = a;
        }
    } else if (b.is_ninf()) {
        result = b;
    } else {
        result = (a.uint() | b.uint()) & DLF::SIGN ? std::max(a.uint(), b.uint()) : std::min(a.uint(), b.uint());
    }

    return result;
} // min

static inline dlfloat16_t
max(const dlfloat16_t &a, const dlfloat16_t &b) {

    dlfloat16_t result;

    if (a.is_ninf()) {
        if (a.is_positive()) {
            result = a;
        } else if (b.is_ninf()) {
            result = b;
        } else {
            result = a;
        }
    } else if (b.is_ninf()) {
        result = b;
    } else {
        result = (a.uint() | b.uint()) & DLF::SIGN ? std::min(a.uint(), b.uint()) : std::max(a.uint(), b.uint());
    }

    return result;
} // max

static inline dlfloat16_t
fma(const dlfloat16_t &a, const dlfloat16_t &b, const dlfloat16_t &c) {

    // fused multiply-add:  a * b + c  with no intermediate DLF rounding

    dlfloat16_t result;

    if (a.is_ninf() || b.is_ninf())
        result = dlfloat16_t(DLF::NINF | ((a.uint() ^ b.uint()) & DLF::SIGN)) + c;
    else if (c.is_ninf())
        result = c;
    else
        result = float(a) * float(b) + float(c);

    return result;
} // fma

// Round float according to DLF rules

static inline float
dlf_round(const float fp) {
    uint32_t    fp32;
    memcpy(&fp32, &fp, sizeof(fp32));

    signed      fp32_biased_exponent = BFGET(fp32, FP32::EXPONENT);

    // Round to nearest with ties away from 0

    uint32_t    fraction = BFGET(fp32, FP32::FRACTION) + DLF::FP32_ROUND;

    if (fraction > FP32::FRACTION) {

        // Rounding caused fraction overflow:
        //                      +------------------- implied 1
        //                      v
        // original fraction    1.111 1111 111a bcde fghi jklm
        // + round              1.000 0000 0010 0000 0000 0000
        // overflows           10.000 0000 000a bcde fghi jklm
        // >> 1                 1.000 0000 0000 abcd efgh ijkl
        // (and adjust exponent)
        // fraction bits to use  .000 0000 00

        fraction = 0;
        fp32_biased_exponent++;
    }

    float       result;

    if (fp32_biased_exponent >= FP32::EXPONENT_BIAS - DLF::EXPONENT_BIAS) {

        if (fp32_biased_exponent < (1 << DLF::EXPONENT_BITS) + FP32::EXPONENT_BIAS - DLF::EXPONENT_BIAS) {

            // Replace (possibly updated) exponent

            BFPUT(fp32, FP32::EXPONENT, fp32_biased_exponent);

            // Truncate FP32 fraction to DLF length

            BFPUT(fp32, FP32::FRACTION, fraction & ~(DLF::FP32_ROUND * 2 - 1));

            memcpy(&result, &fp32, sizeof(result));
        } else {

            // Too large for DLFloat; this also covers FP32 infinity and NaN

            result = fp32 & FP32::SIGN ? -NAN : +NAN;
        }
    } else {

        // Too small for DLFloat, zero exponent and fraction but keep sign

        fp32 &= FP32::SIGN;

        memcpy(&result, &fp32, sizeof(result));
    }

    return result;
} // dlf_round

class float16_t {               // IEEE FP16

private:
    uint16_t    item;           // internal representation (big-endian)

public:
    // Construct big-endian float16_t from integer

    float16_t(const int binary = 0): item(htons(binary)) {}

    // Retrieve integer value, regardless of endianness

    uint16_t uint() const {
        return ntohs(this->item);
    }

    // zDNN ADDED BEGIN -----------------

    // Convert float to float16_t, almost 100% copy-n-paste of dlfloat16_t()'s convert()

    void convert(const float &fp,
                 unsigned    *vic = nullptr) {
        if (vic)
            *vic = 0;

        uint32_t    fp32;
        memcpy(&fp32, &fp, sizeof(fp32));

        signed      fp16_biased_exponent = BFGET(fp32, FP32::EXPONENT) - FP32::EXPONENT_BIAS + FP16::EXPONENT_BIAS;

        // Round to nearest with ties away from 0

        uint32_t    fraction = BFGET(fp32, FP32::FRACTION) + FP16::FP32_ROUND;

        if (fraction > FP32::FRACTION) {

            // Rounding caused fraction overflow:
            //                      +------------------- implied 1
            //                      v
            // original fraction    1.111 1111 111a bcde fghi jklm
            // + round              1.000 0000 0010 0000 0000 0000
            // overflows           10.000 0000 000a bcde fghi jklm
            // >> 1                 1.000 0000 0000 abcd efgh ijkl
            // (and adjust exponent)
            // fraction bits to use  .000 0000 00

            fraction = 0;
            fp16_biased_exponent++;
        }

        // Initialize result with sign

        uint16_t    uint = FP16::SIGN * BFGET(fp32, FP32::SIGN);

        if (fp16_biased_exponent >= 0) {

            if (fp16_biased_exponent < (1 << FP16::EXPONENT_BITS)) {

                // Insert exponent, adjusted to bias for FP16

                BFPUT(uint, FP16::EXPONENT, fp16_biased_exponent);

                // Insert leftmost part of BFP fraction

                BFPUT(uint, FP16::FRACTION, fraction >> (FP32::FRACTION_BITS - FP16::FRACTION_BITS));
            } else {

                // Too large for FP16; this also covers BFP infinity and NaN

                uint |= FP16::NINF;

                if (vic) {
                    if ((fp32 & FP32::EXPONENT) == FP32::EXPONENT) {
                        if (fp32 & FP32::FRACTION) {

                            // BFP NaN

                            *vic = VIC_INVALID;
                        } else {

                            // BFP infinity

                            *vic = VIC_OVERFLOW;
                        }
                    }
                }
            }
        } else {

            // Too small for FP16, leave exponent and fraction zero but keep sign

            if (vic)
                *vic = VIC_UNDERFLOW;
        }

        *this = uint;
    }

    // Construct float16_t from float

    float16_t(const float &fp) { this->convert(fp); } // dlfloat16_t

    // Construct float16_t from double (e.g. constants)

    float16_t(const double &fp) { this->convert(float(fp)); } // dlfloat16_t

    // zDNN ADDED END -----------------

    // Test for +/- 0

    bool is_zero() const {
        return (this->uint() & ~FP16::SIGN) == 0;
    } // is_zero

    // Test sign bit

    bool is_positive() const {
        return (this->uint() & FP16::SIGN) == 0;
    } // is_positive

    // Test for NaN

    bool is_nan() const {
        return (this->uint() & ~FP16::SIGN) > FP16::EXPONENT;
    } // is_nan

    // Test for SNaN

    bool is_signaling() const {
        return this->is_nan() && (this->uint() & FP16::FRACTION) <= (FP16::FRACTION >> 1);
    } // is_signaling

    // Test for infinity

    bool is_infinity() const {
        return (this->uint() & ~FP16::SIGN) == FP16::EXPONENT;
    } // is_infinity

    // Convert dlfloat16_t -> float16_t (used by vector instruction)

    void convert(const dlfloat16_t &dlf,
                 unsigned         *vic = nullptr) {

        if (vic)
            *vic = 0;

        uint16_t    fp16;

        if (dlf.is_ninf()) {

            // Sign is not propagated from NINF to NaN

            fp16 = FP16::QNAN;

            if (vic)
                *vic = VIC_INVALID;
        } else {

            // Initialize result with sign

            fp16 = FP16::SIGN * BFGET(dlf.uint(), DLF::SIGN);

            if (!dlf.is_zero()) {

                signed  fp16_biased_exponent = BFGET(dlf.uint(), DLF::EXPONENT) - DLF::EXPONENT_BIAS + FP16::EXPONENT_BIAS;

                if (fp16_biased_exponent > 0) {

                    if (fp16_biased_exponent < (1 << FP16::EXPONENT_BITS) - 1) {

                        // Insert exponent, adjusted to bias for FP16

                        BFPUT(fp16, FP16::EXPONENT, fp16_biased_exponent);

                        // Insert fraction, left adjusted

                        BFPUT(fp16, FP16::FRACTION, BFGET(dlf.uint(), DLF::FRACTION) << (FP16::FRACTION_BITS - DLF::FRACTION_BITS));
                    } else {

                        // Too large for FP16

                        fp16 |= FP16::INF;

                        if (vic)
                            *vic = VIC_OVERFLOW;
                    }
                } else if (fp16_biased_exponent == 0) {

                    // Subnormal number here, so implied leading 1 must be prefixed explicitly

                    uint16_t    fraction = (1 << DLF::FRACTION_BITS) + BFGET(dlf.uint(), DLF::FRACTION);

                    // Insert extended fraction

                    BFPUT(fp16, FP16::FRACTION, fraction << (FP16::FRACTION_BITS - DLF::FRACTION_BITS - 1));
                } else {

                    // Too small for FP16, consider it (signed) zero

                    if (vic)
                        *vic = VIC_UNDERFLOW;
                }
            }
        }

        *this = fp16;
    } // float16_t

    // Construct float16_t <- dlfloat16_t (used by vector instruction)

    float16_t(const dlfloat16_t &dlf) {
        this->convert(dlf);
    } // float16_t

    // Convert float16_t -> dlfloat16_t (used by vector instruction)

    dlfloat16_t convert(unsigned *vic = nullptr) const {

        if (vic)
            *vic = 0;

        // Initialize result with sign

        uint16_t    dlf = DLF::SIGN * BFGET(this->uint(), FP16::SIGN);

        if (this->is_infinity()) {

            // Sign is propagated from infinity to NINF

            dlf |= DLF::NINF;

            if (vic)
                *vic = VIC_OVERFLOW;
        } else if (this->is_nan()) {

            // Sign is propagated from NaN to NINF

            dlf |= DLF::NINF;

            if (vic)
                *vic = VIC_INVALID;
        } else if (!this->is_zero()) {

            signed      exponent = BFGET(this->uint(), FP16::EXPONENT);
            uint32_t    fraction = BFGET(this->uint(), FP16::FRACTION);

            if (exponent == 0) {    // subnormal FP16 number

                // Normalize fraction

                unsigned    normalize  = __builtin_clz(fraction) - 16 - FP16::EXPONENT_BITS;

                fraction <<= normalize;

                // Chop leading 1 - this is now a normal number

                fraction  &= FP16::FRACTION;

                // Adjust exponent accordingly

                exponent = FP16::FRACTION_BITS - DLF::FRACTION_BITS - normalize;
            }

            // Round fraction with ties away from 0 and shorten to DLF length

            fraction = (fraction + FP16::DLF_ROUND) >> (FP16::FRACTION_BITS - DLF::FRACTION_BITS);
            if (fraction > DLF::FRACTION) {
                fraction = 0;
                exponent++;
            }

            // Insert fraction

            BFPUT(dlf, DLF::FRACTION, fraction);

            // Insert exponent, biased to DLF

            BFPUT(dlf, DLF::EXPONENT, exponent - FP16::EXPONENT_BIAS + DLF::EXPONENT_BIAS);
        }

        return dlf;
    } // dlfloat16_t

    operator dlfloat16_t() const {
        return this->convert();
    } // operator dlfloat16_t

    // Convert float16_t to float (for display)

    operator float() const {

        float       fp;

        if (this->is_zero()) {
            fp = this->is_positive() ? +0.0f : -0.0f;
        } else if (this->is_infinity()) {
            fp = this->is_positive() ? +INFINITY : -INFINITY;
        } else if (this->is_nan()) {
            fp = this->is_positive() ? +NAN : -NAN;
        } else {

            uint32_t    fp32     = FP32::SIGN * BFGET(this->uint(), FP16::SIGN);
            signed      exponent = BFGET(this->uint(), FP16::EXPONENT);
            uint32_t    fraction = BFGET(this->uint(), FP16::FRACTION);

            if (exponent > 0) {     // normal FP16 number

                // Adjust fraction to FP32 length

                fraction <<= FP32::FRACTION_BITS - FP16::FRACTION_BITS;

            } else {                // subnormal FP16 number

                // Adjust fraction such that the leading 1 is chopped - this is now a normal number

                unsigned    normalize = __builtin_clz(fraction) - FP32::EXPONENT_BITS;

                fraction <<= normalize;

                // Adjust exponent accordingly

                exponent = FP32::FRACTION_BITS + 1 - FP16::FRACTION_BITS - normalize;
            }

            // Insert fraction

            BFPUT(fp32, FP32::FRACTION, fraction);

            // Insert exponent, biased to FP32

            BFPUT(fp32, FP32::EXPONENT, exponent - FP16::EXPONENT_BIAS + FP32::EXPONENT_BIAS);

            memcpy(&fp, &fp32, sizeof(fp));
        }

        return fp;
    } // operator float

}; // class float16_t

#if __cplusplus < 201103
#undef constexpr
#undef nullptr
#endif
#endif
