/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- SmallFP.hpp -----------------------------===//
//
// 8 and 16 bits floating point types.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SMALL_FP_H
#define ONNX_MLIR_SMALL_FP_H

#include "mlir/IR/BuiltinAttributes.h"
#include "src/Support/SmallFPConversion.h"
#include "llvm/ADT/APFloat.h"

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
      : SmallFPBase(fromFloat(static_cast<float>(x))) {}

  // Support static_cast<T>(*this) for any T that float converts to.
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP>>>
  explicit operator T() const {
    return static_cast<float>(toFloat());
  }

  bool isNaN() const {
    // Down cast in case FP overrides isNaNImpl().
    FP fp = FP::bitcastFromUInt(ui);
    return fp.isNaNImpl();
  }

  // Same as static_cast<float>(*this).
  float toFloat() const {
    // Down cast in case FP overrides toFloatImpl().
    FP fp = FP::bitcastFromUInt(ui);
    return fp.toFloatImpl();
  }

  llvm::APFloat toAPFloat() const;

  // Same as bitcast<bitcasttype>(*this).
  constexpr bitcasttype bitcastToUInt() const { return ui; }

  // Same as static_cast<FP>(f).
  static FP fromFloat(float f) { return FP::fromFloatImpl(f); }

  static FP fromAPFloat(llvm::APFloat a);

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

protected:
  float isNaNImpl() const { return toAPFloat().isNaN(); }

  float toFloatImpl() const { return toAPFloat().convertToFloat(); }

  static FP fromFloatImpl(float f) { return fromAPFloat(llvm::APFloat(f)); }

private:
  bitcasttype ui;
};

extern template class SmallFPBase<float_16, 16>;
extern template class SmallFPBase<bfloat_16, 16>;
extern template class SmallFPBase<float_8e4m3fn, 8>;
extern template class SmallFPBase<float_8e4m3fnuz, 8>;
extern template class SmallFPBase<float_8e5m2, 8>;
extern template class SmallFPBase<float_8e5m2fnuz, 8>;

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
  static constexpr float max = 65504.0f;

protected:
  friend class detail::SmallFPBase<float_16, 16>;
  float isNaNImpl() const {
    uint16_t u16 = bitcastToUInt();
    return (u16 & 0x7C00) == 0x7C00 && (u16 & 0x03FF) != 0;
  }
  float toFloatImpl() const { return om_f16_to_f32(bitcastToUInt()); }
  static float_16 fromFloatImpl(float f) {
    return bitcastFromUInt(om_f32_to_f16(f));
  }
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
  static constexpr float max = 3.38953139e38f; // ldexpf(255.0, 120)

protected:
  friend class detail::SmallFPBase<bfloat_16, 16>;
  float isNaNImpl() const {
    uint16_t u16 = bitcastToUInt();
    return (u16 & 0x7F80) == 0x7F80 && (u16 & 0x007F) != 0;
  }
  float toFloatImpl() const { return om_bf16_to_f32(bitcastToUInt()); }
  static bfloat_16 fromFloatImpl(float f) {
    return bitcastFromUInt(om_f32_to_bf16(f));
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
  static constexpr float max = 448.0f;
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
  static constexpr float max = 240.0f;
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
  static constexpr float max = 57344.0f;
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
  static constexpr float max = 57344.0f;
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
#endif
