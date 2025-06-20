/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- Int4.hpp -----------------------------===//
//
// 4 bits integer type
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_INT_4_H
#define ONNX_MLIR_INT_4_H

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace onnx_mlir {

namespace detail {

template <typename IT>
class Int4Base {
public:
  using StorageType = uint8_t;

  constexpr Int4Base() : ui() {}
  constexpr explicit Int4Base(const IT &it) : ui(it.ui) {}

  /// Extracts the 4-bit integer from a packed representation.
  /// The packed representation is expected to be packed in the LSB byte in the
  /// integer T. If `isFirst` is true, it extracts the LSB 4 bits; otherwise,
  /// it extracts the MSB 4 bits from this byte.
  template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>>>
  static constexpr IT extractFromPacked(T packed, bool isFirst) {
    IT result;
    if (isFirst) {
      result.ui = static_cast<StorageType>(packed & 0x0F);
    } else {
      result.ui = static_cast<StorageType>((packed >> 4) & 0x0F);
    }
    return result;
  }

  bool operator==(IT other) const { return ui == other.ui; }

  bool operator!=(IT other) const { return !(*this == other); }

protected:
  StorageType ui;
};

} // namespace detail

template <class T>
constexpr bool isAnyInt4Type = std::is_base_of_v<detail::Int4Base<T>, T>;

class int_4 : public detail::Int4Base<int_4> {
  using Base = detail::Int4Base<int_4>;

public:
  using Base::Int4Base;

  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, int_4>>>
  explicit int_4(T value) {
    auto expanded = static_cast<int64_t>(value);
    ui = expanded & 0x0F;
  }

  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, int_4>>>
  explicit operator T() const {
    if (ui & 0x08) {
      // If the sign bit is set, extend the sign.
      return static_cast<T>(static_cast<int8_t>(ui | 0xF0));
    }
    return static_cast<T>(ui);
  }
};
static_assert(sizeof(int_4) == 1, "int_4 must be 1 byte in size");

class uint_4 : public detail::Int4Base<uint_4> {
  using Base = detail::Int4Base<uint_4>;

public:
  using Base::Int4Base;

  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, uint_4>>>
  explicit uint_4(T value) {
    const auto expanded = static_cast<uint64_t>(value);
    ui = expanded & 0x0F;
  }

  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, uint_4>>>
  explicit operator T() const {
    return static_cast<T>(ui);
  }
};
static_assert(sizeof(uint_4) == 1, "int_4 must be 1 byte in size");

} // namespace onnx_mlir

// Enable DenseElementsAttr to operate on int_4 and uint_4 types and enables
// type dependent template programming
template <>
struct std::numeric_limits<onnx_mlir::int_4> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = true;
};

template <>
struct std::numeric_limits<onnx_mlir::uint_4> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = false;
};
#endif
