/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- FloatingPoint16.hpp --------------------------===//
//
// 16 bit floating point types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/APFloat.h"

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
  constexpr explicit FP16Base(FP16 f16) : u16(f16.u16) {}
  // Support static_cast<FP16>(X) for any x that is convertible to float.
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP16>>>
  explicit FP16Base(T x)
      : u16(fromAPFloat(llvm::APFloat(static_cast<float>(x))).u16) {}

  // Support static_cast<T>(*this) for any T that float converts to.
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP16>>>
  explicit operator T() const {
    return static_cast<float>(toFloat());
  }

  llvm::APFloat toAPFloat() const;

  // Same as static_cast<float>(*this).
  float toFloat() const { return toAPFloat().convertToFloat(); }

  // Substitute for reinterpret_cast<uint16_t>(*this), which C++ doesn't allow.
  constexpr bitcasttype bitcastToU16() const { return u16; }

  static FP16 fromAPFloat(llvm::APFloat a);

  // Substitute for reinterpret_cast<FP16>(f), which C++ doesn't allow.
  static FP16 fromFloat(float f) { return fromAPFloat(llvm::APFloat(f)); }

  // Substitute for reinterpret_cast<FP16>(u), which C++ doesn't allow.
  static constexpr FP16 bitcastFromU16(bitcasttype u) {
    FP16 f16;
    f16.u16 = u;
    return f16;
  }

private:
  bitcasttype u16;
};

extern template class FP16Base<float_16>;
extern template class FP16Base<bfloat_16>;

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

// When doing arithmetic on a template type T that may be a (b)float_16 or a
// native arithmetic type, convert arguments to toArithmetic<T> and back, e.g.:
//
//   template <typename T> T Sqrt(T lhs, T rhs) {
//     return static_cast<T>(sqrtf(static_cast<toArithmetic<T>>(x)));
//   }
//
template <typename T>
using toArithmetic = std::conditional_t<isFP16Type<T>, float, T>;

} // namespace onnx_mlir