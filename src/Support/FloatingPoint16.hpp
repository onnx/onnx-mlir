/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- FloatingPoint16.hpp --------------------------===//
//
// 16 bit floating point types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"

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
// clang-format off
// On x86-64 build config -DCMAKE_CXX_FLAGS=-march=native defines __F16C__.
#if defined(__x86_64__) && defined(__F16C__)
// https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-9/details-about-intrinsics-for-half-floats.html
#include <immintrin.h>
#define FLOAT16_TO_FLOAT32(u16) _cvtsh_ss(u16)
#define FLOAT32_TO_FLOAT16(f32) _cvtss_sh(f32, /*ROUND TO NEAREST EVEN*/ 0)
#endif
// On MacBook Pro no build config is needed to define __ARM_FP16_FORMAT_IEEE.
#if defined(__ARM_FP16_FORMAT_IEEE)
// https://arm-software.github.io/acle/main/acle.html#half-precision-floating-point
#define FLOAT16_TO_FLOAT32(u16) static_cast<float>(detail::bitcast<__fp16>(u16))
#define FLOAT32_TO_FLOAT16(f32) detail::bitcast<uint16_t>(static_cast<__fp16>(f32))
#endif
// clang-format on

namespace onnx_mlir {

class float_16;
class bfloat_16;

namespace detail {

// Base class for float_16, bfloat_16.
template <typename FP16> // FP16 is the derived class, float_16 or bfloat_16.
class FP16Base {
public:
  using bitcasttype = uint16_t;

  constexpr FP16Base() : u16() {}
  constexpr explicit FP16Base(const FP16 &f16) : u16(f16.u16) {}
  // Support static_cast<FP16>(X) for any x that is convertible to float.
  // Use FP16::fromFloat() in case FP16 overrides fromFloat().
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP16>>>
  explicit FP16Base(const T &x)
      : FP16Base(FP16::fromFloat(static_cast<float>(x))) {}

  // Support static_cast<T>(*this) for any T that float converts to.
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP16>>>
  explicit operator T() const {
    // Down cast in case FP16 overrides FP16::toFloat().
    FP16 fp = FP16::bitcastFromU16(u16);
    return static_cast<float>(fp.toFloat());
  }

  llvm::APFloat toAPFloat() const;

  // Same as static_cast<float>(*this).
  float toFloat() const { return toAPFloat().convertToFloat(); }

  // Same as bitcast<uint16_t>(*this).
  constexpr bitcasttype bitcastToU16() const { return u16; }

  static FP16 fromAPFloat(llvm::APFloat a);

  // Same as static_cast<FP16>(f).
  static FP16 fromFloat(float f) { return fromAPFloat(llvm::APFloat(f)); }

  // Same as bitcast<FP16>(u).
  static constexpr FP16 bitcastFromU16(bitcasttype u) {
    FP16 f16;
    f16.u16 = u;
    return f16;
  }

  // Almost the same as u16 == other.u16, except
  // * 0 and minus 0 (0x0 and 0x8000) are also equated
  // * NaN values are not equal to themselves
  bool operator==(FP16 other) const { return toFloat() == other.toFloat(); }

  bool operator!=(FP16 other) const { return !(*this == other); }

private:
  bitcasttype u16;
};

extern template class FP16Base<float_16>;
extern template class FP16Base<bfloat_16>;

// TODO: Replace with std::bit_cast in C++20.
template <class To, class From>
To bitcast(From x) {
  return *reinterpret_cast<To *>(&x);
}

} // namespace detail

template <class T>
inline constexpr bool isFP16Type = std::is_base_of_v<detail::FP16Base<T>, T>;

// Represents a FLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
class float_16 : public detail::FP16Base<float_16> {
  using Base = detail::FP16Base<float_16>;

public:
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::IEEEhalf();
  }
#if defined(FLOAT16_TO_FLOAT32)
  float toFloat() const { return FLOAT16_TO_FLOAT32(bitcastToU16()); }
#endif
#if defined(FLOAT32_TO_FLOAT16)
  static float_16 fromFloat(float f) {
    return bitcastFromU16(FLOAT32_TO_FLOAT16(f));
  }
#endif
};
static_assert(sizeof(float_16) * CHAR_BIT == 16, "float_16 is 16 bits wide");

// Represents a BFLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
class bfloat_16 : public detail::FP16Base<bfloat_16> {
  using Base = detail::FP16Base<bfloat_16>;

public:
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::BFloat();
  }
};
static_assert(sizeof(bfloat_16) * CHAR_BIT == 16, "bfloat_16 is 16 bits wide");

} // namespace onnx_mlir

// Enable DenseElementsAttr to operate on float_16, bfloat_16 data types.
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::float_16> {
  static constexpr bool value = true;
};
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::bfloat_16> {
  static constexpr bool value = true;
};
