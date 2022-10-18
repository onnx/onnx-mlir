/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- DType.hpp ------------------------------===//
//
// Basic data types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/ErrorHandling.h"

namespace onnx_mlir {

// Numerical representation of basic data types.
//
// DTYPE faithfully copies onnx::TensorProto::DataType from
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
// and DTYPE and onnx::TensorProto::DataType can be used interchangeably.
// In some places it is convenient to use DTYPE to avoid compile time
// dependencies on third_party/onnx.
namespace DTYPE {
enum DataType : int {
  // clang-format off
  UNDEFINED = 0,
  // Basic types.
  FLOAT = 1,   // float
  UINT8 = 2,   // uint8_t
  INT8 = 3,    // int8_t
  UINT16 = 4,  // uint16_t
  INT16 = 5,   // int16_t
  INT32 = 6,   // int32_t
  INT64 = 7,   // int64_t
  STRING = 8,  // string
  BOOL = 9,    // bool

  // IEEE754 half-precision floating-point format (16 bits wide).
  // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  FLOAT16 = 10,

  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,     // complex with float32 real and imaginary components
  COMPLEX128 = 15,    // complex with float64 real and imaginary components

  // Non-IEEE floating-point format based on IEEE754 single-precision
  // floating-point number truncated to 16 bits.
  // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
  BFLOAT16 = 16
  // clang-format on
};
}

// Helper functions.
float U16ToF32(uint16_t);
uint16_t F32ToU16(float);

template <int TY>
struct DType {
  static constexpr int dtype = TY;
};

#define DEFINE_DType_X(TY, CPPTY, XTRA)                                        \
  template <>                                                                  \
  struct DType<DTYPE::TY> {                                                    \
    static constexpr int dtype = DTYPE::TY;                                    \
    using type = CPPTY;                                                        \
    XTRA                                                                       \
  }

#define DEFINE_DType(TY, CPPTY)                                                \
  DEFINE_DType_X(                                                              \
      TY, CPPTY, using unpacked_type = type;                                   \
      static type pack(unpacked_type unpacked) {                               \
        return unpacked;                                                       \
      } static unpacked_type unpack(type packed) { return packed; })

DEFINE_DType(FLOAT, float);
DEFINE_DType(DOUBLE, double);
DEFINE_DType(BOOL, bool);
DEFINE_DType(INT8, int8_t);
DEFINE_DType(UINT8, uint8_t);
DEFINE_DType(INT16, int16_t);
DEFINE_DType(UINT16, uint16_t);
DEFINE_DType(INT32, int32_t);
DEFINE_DType(UINT32, uint32_t);
DEFINE_DType(INT64, int64_t);
DEFINE_DType(UINT64, uint64_t);
DEFINE_DType_X(
    FLOAT16, uint16_t, using unpacked_type = float;
    static type pack(unpacked_type unpacked) {
      return F32ToU16(unpacked);
    } static unpacked_type unpack(type packed) { return U16ToF32(packed); });

#if 0
template <template <typename, typename...> class Action,
    typename Out, typename... Ts>
struct dispatchIntOrFP {
  static Out eval(mlir::Type type, Ts... xs) {
#define ACT(TY) (Action<DType<DTYPE::TY>, Ts...>::eval(xs...))
    // clang-format off
    if (type.isBF16()) llvm_unreachable("BF16 is unsupported");
    if (type.isF16()) return ACT(FLOAT16);
    if (type.isF32()) return ACT(FLOAT);
    if (type.isF64()) return ACT(DOUBLE);
    auto itype = type.cast<mlir::IntegerType>();
    switch (itype.getWidth()) {
      case  1: return ACT(BOOL);
      case  8: return itype.isUnsigned() ? ACT(UINT8)  : ACT(INT8);
      case 16: return itype.isUnsigned() ? ACT(UINT16) : ACT(INT16);
      case 32: return itype.isUnsigned() ? ACT(UINT32) : ACT(INT32);
      case 64: return itype.isUnsigned() ? ACT(UINT64) : ACT(INT64);
      default: llvm_unreachable("unsupported integer width");
    }
    // clang-format on
#undef ACT
  }
};
#endif

template <template <typename, typename...> class Action, typename Out,
    typename... Ts>
struct dispatchInt {
  static Out eval(mlir::Type type, Ts... xs) {
#define ACT(TY) (Action<DType<DTYPE::TY>, Ts...>::eval(xs...))
    // clang-format off
    auto itype = type.cast<mlir::IntegerType>();
    switch (itype.getWidth()) {
      case  1: return ACT(BOOL);
      case  8: return itype.isUnsigned() ? ACT(UINT8)  : ACT(INT8);
      case 16: return itype.isUnsigned() ? ACT(UINT16) : ACT(INT16);
      case 32: return itype.isUnsigned() ? ACT(UINT32) : ACT(INT32);
      case 64: return itype.isUnsigned() ? ACT(UINT64) : ACT(INT64);
      default: llvm_unreachable("unsupported integer width");
    }
    // clang-format on
#undef ACT
  }
};

template <template <typename, typename...> class Action, typename Alt,
    typename Out, typename... Ts>
struct dispatchFPOr {
  static Out eval(mlir::Type type, Ts... xs) {
#define ACT(TY) (Action<DType<DTYPE::TY>, Ts...>::eval(xs...))
    // clang-format off
    if (type.isBF16()) llvm_unreachable("BF16 is unsupported");
    if (type.isF16()) return ACT(FLOAT16);
    if (type.isF32()) return ACT(FLOAT);
    if (type.isF64()) return ACT(DOUBLE);
    return Alt::eval(type, xs...);
    // clang-format on
#undef ACT
  }
};

template <typename Out, typename... Ts>
struct dispatchFail {
  static Out eval(mlir::Type type, Ts... xs) {
    llvm_unreachable("unsupported type");
  }
};

template <template <typename, typename...> class Action, typename Out,
    typename... Ts>
using dispatchFP = dispatchFPOr<Action, dispatchFail<Out, Ts...>, Out, Ts...>;

template <template <typename, typename...> class Action, typename Out,
    typename... Ts>
using dispatchFPOrInt =
    dispatchFPOr<Action, dispatchInt<Action, Out, Ts...>, Out, Ts...>;

// Helper functions frequently used together with dispatch classes.

template <typename New, typename Old = char>
llvm::ArrayRef<New> castArrayRef(llvm::ArrayRef<Old> a) {
  return llvm::makeArrayRef(reinterpret_cast<const New *>(a.data()),
      (a.size() * sizeof(Old)) / sizeof(New));
}

template <typename New, typename Old = char>
llvm::MutableArrayRef<New> castMutableArrayRef(llvm::MutableArrayRef<Old> a) {
  return llvm::makeMutableArrayRef(reinterpret_cast<New *>(a.data()),
      (a.size() * sizeof(Old)) / sizeof(New));
}

template <typename Src, typename Dst, typename Fn>
void fillOrTransform(
    llvm::ArrayRef<Src> src, llvm::MutableArrayRef<Dst> dst, Fn fn) {
  if (src.size() == 1)
    std::fill(dst.begin(), dst.end(), fn(src.front()));
  else
    std::transform(src.begin(), src.end(), dst.begin(), fn);
}

template <typename U>
using onlyFP = std::enable_if_t<std::is_floating_point_v<U>>;

template <typename U>
using onlyNumber = std::enable_if_t<!std::is_same_v<U, bool>>;

} // namespace onnx_mlir