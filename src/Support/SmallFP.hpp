/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- SmallFP.hpp -----------------------------===//
//
// 8 and 16 bits floating point types.
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
class float_8e4m3fn;
class float_8e4m3fnuz;
class float_8e5m2;
class float_8e5m2fnuz;

namespace detail {

// Base class for float_16, bfloat_16, and float_8 classes.
// FP is the derived class (float_16 etc) and UInt is uint16_t or uint8_t.
template <typename FP, unsigned BITWIDTH>
class SmallFPBase {
public:
  using bitcasttype = std::conditional_t<BITWIDTH == 8, uint8_t,
      std::conditional_t<BITWIDTH == 16, uint16_t, void>>;

  constexpr SmallFPBase() : ui() {}
  constexpr explicit SmallFPBase(const FP &fp) : ui(fp.ui) {}
  // Support static_cast<FP>(X) for any x that is convertible to float.
  // Use FP::fromFloat() in case FP overrides fromFloat().
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP>>>
  explicit SmallFPBase(const T &x)
      : SmallFPBase(FP::fromFloat(static_cast<float>(x))) {}

  // Support static_cast<T>(*this) for any T that float converts to.
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP>>>
  explicit operator T() const {
    // Down cast in case FP overrides FP::toFloat().
    FP fp = FP::bitcastFromUInt(ui);
    return static_cast<float>(fp.toFloat());
  }

  llvm::APFloat toAPFloat() const;

  // Same as static_cast<float>(*this).
  float toFloat() const { return toAPFloat().convertToFloat(); }

  // Same as bitcast<bitcasttype>(*this).
  constexpr bitcasttype bitcastToUInt() const { return ui; }

  static FP fromAPFloat(llvm::APFloat a);

  // Same as static_cast<FP>(f).
  static FP fromFloat(float f) { return fromAPFloat(llvm::APFloat(f)); }

  // Same as bitcast<FP>(u).
  static constexpr FP bitcastFromUInt(bitcasttype u) {
    FP fp;
    fp.ui = u;
    return fp;
  }

  // Almost the same as ui == other.ui, except
  // * 0 and minus 0 are also equated
  // * NaN values are not equal to themselves
  bool operator==(FP other) const { return toFloat() == other.toFloat(); }

  bool operator!=(FP other) const { return !(*this == other); }

private:
  bitcasttype ui;
};

extern template class SmallFPBase<float_16, 16>;
extern template class SmallFPBase<bfloat_16, 16>;
extern template class SmallFPBase<float_8e4m3fn, 8>;
extern template class SmallFPBase<float_8e4m3fnuz, 8>;
extern template class SmallFPBase<float_8e5m2, 8>;
extern template class SmallFPBase<float_8e5m2fnuz, 8>;

// TODO: Replace with std::bit_cast in C++20.
template <class To, class From>
To bitcast(From x) {
  return *reinterpret_cast<To *>(&x);
}

} // namespace detail

template <class T>
inline constexpr bool isSmallFPType =
    std::is_base_of_v<detail::SmallFPBase<T, sizeof(T) * CHAR_BIT>, T>;

// Represents a FLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
class float_16 : public detail::SmallFPBase<float_16, 16> {
  using Base = detail::SmallFPBase<float_16, 16>;

public:
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::IEEEhalf();
  }
#if defined(FLOAT16_TO_FLOAT32)
  float toFloat() const { return FLOAT16_TO_FLOAT32(bitcastToUInt()); }
#endif
#if defined(FLOAT32_TO_FLOAT16)
  static float_16 fromFloat(float f) {
    return bitcastFromUInt(FLOAT32_TO_FLOAT16(f));
  }
#endif
};
static_assert(sizeof(float_16) * CHAR_BIT == 16, "float_16 is 16 bits wide");

// Represents a BFLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
class bfloat_16 : public detail::SmallFPBase<bfloat_16, 16> {
  using Base = detail::SmallFPBase<bfloat_16, 16>;

public:
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::BFloat();
  }
};
static_assert(sizeof(bfloat_16) * CHAR_BIT == 16, "bfloat_16 is 16 bits wide");

// Represents a F8E4M3FN value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint8_t, float, etc.
class float_8e4m3fn : public detail::SmallFPBase<float_8e4m3fn, 8> {
  using Base = detail::SmallFPBase<float_8e4m3fn, 8>;

public:
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::Float8E4M3FN();
  }
};
static_assert(
    sizeof(float_8e4m3fn) * CHAR_BIT == 8, "float_8e4m3fn is 8 bits wide");

// Represents a F8E4M3FNUZ value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint8_t, float, etc.
class float_8e4m3fnuz : public detail::SmallFPBase<float_8e4m3fnuz, 8> {
  using Base = detail::SmallFPBase<float_8e4m3fnuz, 8>;

public:
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::Float8E4M3FNUZ();
  }
};
static_assert(
    sizeof(float_8e4m3fnuz) * CHAR_BIT == 8, "float_8e4m3fnuz is 8 bits wide");

// Represents a F8E5M2 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint8_t, float, etc.
class float_8e5m2 : public detail::SmallFPBase<float_8e5m2, 8> {
  using Base = detail::SmallFPBase<float_8e5m2, 8>;

public:
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::Float8E5M2();
  }
};
static_assert(
    sizeof(float_8e5m2) * CHAR_BIT == 8, "float_8e5m2 is 8 bits wide");

// Represents a F8E5M2FNUZ value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint8_t, float, etc.
class float_8e5m2fnuz : public detail::SmallFPBase<float_8e5m2fnuz, 8> {
  using Base = detail::SmallFPBase<float_8e5m2fnuz, 8>;

public:
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::Float8E5M2FNUZ();
  }
};
static_assert(
    sizeof(float_8e5m2fnuz) * CHAR_BIT == 8, "float_8e5m2fnuz is 8 bits wide");

} // namespace onnx_mlir

// Enable DenseElementsAttr to operate on float_16, bfloat_16, and float_8s.
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::float_16> {
  static constexpr bool value = true;
};
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::bfloat_16> {
  static constexpr bool value = true;
};
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::float_8e4m3fn> {
  static constexpr bool value = true;
};
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<
    onnx_mlir::float_8e4m3fnuz> {
  static constexpr bool value = true;
};
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::float_8e5m2> {
  static constexpr bool value = true;
};
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<
    onnx_mlir::float_8e5m2fnuz> {
  static constexpr bool value = true;
};
