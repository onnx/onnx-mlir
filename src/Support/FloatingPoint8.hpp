/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- FloatingPoint8.hpp -------------------------===//
//
// 8 bit floating point types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"

namespace onnx_mlir {

class float_8e4m3fn;
class float_8e4m3fnuz;
class float_8e5m2;
class float_8e5m2fnuz;

namespace detail {

// Base class for float_8 classes.
template <typename FP8> // FP8 is the derived class, float_8e4m3fn etc.
class FP8Base {
public:
  using bitcasttype = uint8_t;

  constexpr FP8Base() : u8() {}
  constexpr explicit FP8Base(const FP8 &f8) : u8(f8.u8) {}
  // Support static_cast<FP8>(X) for any x that is convertible to float.
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP8>>>
  explicit FP8Base(const T &x)
      : u8(fromAPFloat(llvm::APFloat(static_cast<float>(x))).u8) {}

  // Support static_cast<T>(*this) for any T that float converts to.
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP8>>>
  explicit operator T() const {
    return static_cast<float>(toFloat());
  }

  llvm::APFloat toAPFloat() const;

  // Same as static_cast<float>(*this).
  float toFloat() const { return toAPFloat().convertToFloat(); }

  // Substitute for reinterpret_cast<uint8_t>(*this), which C++ doesn't allow.
  constexpr bitcasttype bitcastToU8() const { return u8; }

  static FP8 fromAPFloat(llvm::APFloat a);

  // Same as static_cast<FP8>(f).
  static FP8 fromFloat(float f) { return fromAPFloat(llvm::APFloat(f)); }

  // Substitute for reinterpret_cast<FP8>(u), which C++ doesn't allow.
  static constexpr FP8 bitcastFromU8(bitcasttype u) {
    FP8 f8;
    f8.u8 = u;
    return f8;
  }

  // Almost the same as u8 == other.u8, except
  // * 0 and minus 0 are also equated
  // * NaN values are not equal to themselves
  bool operator==(FP8 other) const { return toFloat() == other.toFloat(); }

  bool operator!=(FP8 other) const { return !(*this == other); }

private:
  bitcasttype u8;
};

extern template class FP8Base<float_8e4m3fn>;
extern template class FP8Base<float_8e4m3fnuz>;
extern template class FP8Base<float_8e5m2>;
extern template class FP8Base<float_8e5m2fnuz>;

} // namespace detail

template <class T>
inline constexpr bool isFP8Type = std::is_base_of_v<detail::FP8Base<T>, T>;

// Represents a F8E4M3FN value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint8_t, float, etc.
class float_8e4m3fn : public detail::FP8Base<float_8e4m3fn> {
  using Base = detail::FP8Base<float_8e4m3fn>;

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
class float_8e4m3fnuz : public detail::FP8Base<float_8e4m3fnuz> {
  using Base = detail::FP8Base<float_8e4m3fnuz>;

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
class float_8e5m2 : public detail::FP8Base<float_8e5m2> {
  using Base = detail::FP8Base<float_8e5m2>;

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
class float_8e5m2fnuz : public detail::FP8Base<float_8e5m2fnuz> {
  using Base = detail::FP8Base<float_8e5m2fnuz>;

public:
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::Float8E5M2FNUZ();
  }
};
static_assert(
    sizeof(float_8e5m2fnuz) * CHAR_BIT == 8, "float_8e5m2fnuz is 8 bits wide");

} // namespace onnx_mlir

// Enable DenseElementsAttr to operate on float_8 data types.
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
