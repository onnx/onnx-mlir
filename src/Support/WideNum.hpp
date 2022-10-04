/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- WideNum.hpp -----------------------------===//
//
// WideNum data type.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Support/DType.hpp"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"

namespace onnx_mlir {

// Union of 64-bit integers and double precision floating point numbers.
// It is tagless and should always be used in a conjunction with a DType.
// The dtype tags which field of the union is populated:
// dbl if isFloat(dtype), i64 or u64 if isSigned/UnsignedInt(dtype).
//
// WideNum satisfies for all cpp types X and Y values x of type X
//
//   static_cast<Y>(x) == WideNum::from<X>(dtype, x).To<Y>(dtype)
//
// provided the wide type of dtype (double, int64_t, uint64_t) has enough
// precision and range to represent x up to the precision and range of Y.
//
union WideNum {
  double dbl;   // Floating point numbers with precision and range up to double.
  int64_t i64;  // Signed ints up to bitwidth 64.
  uint64_t u64; // Unsigned ints up to bitwidth 64, including bool.

  llvm::APFloat toAPFloat(DType tag) const;

  static WideNum fromAPFloat(DType tag, llvm::APFloat x);

  llvm::APInt toAPInt(DType tag) const;

  static WideNum fromAPInt(DType tag, llvm::APInt x);

  template <typename T>
  constexpr T to(DType dtag) const {
    switch (dtag) {
    case DType::BOOL:
    case DType::UINT8:
    case DType::UINT16:
    case DType::UINT32:
    case DType::UINT64:
      return static_cast<T>(u64);
    case DType::INT8:
    case DType::INT16:
    case DType::INT32:
    case DType::INT64:
      return static_cast<T>(i64);
    case DType::DOUBLE:
    case DType::FLOAT:
    case DType::FLOAT16:
    case DType::BFLOAT16:
      return static_cast<T>(dbl);
    default:
      llvm_unreachable("to unsupported dtype");
    }
  }

  template <typename T>
  static constexpr WideNum from(DType dtag, T x) {
    switch (dtag) {
    case DType::BOOL:
    case DType::UINT8:
    case DType::UINT16:
    case DType::UINT32:
    case DType::UINT64:
      return {.u64 = static_cast<uint64_t>(x)};
    case DType::INT8:
    case DType::INT16:
    case DType::INT32:
    case DType::INT64:
      return {.i64 = static_cast<int64_t>(x)};
    case DType::DOUBLE:
    case DType::FLOAT:
    case DType::FLOAT16:
    case DType::BFLOAT16:
      return {.dbl = static_cast<double>(x)};
    default:
      llvm_unreachable("from unsupported dtype");
    }
  }

  void store(DType dtag, llvm::MutableArrayRef<char> memory) const;

  static WideNum load(DType dtag, llvm::ArrayRef<char> memory);
};
static_assert(sizeof(WideNum) * CHAR_BIT == 64, "WideNum is 64 bits wide");

template <DType DTYPE>
struct WideDType {
  using narrowtype = CppType<DTYPE>;
  using type = typename DTypeTrait<DTYPE>::widetype;
  static constexpr DType dtype = toDType<type>;
  static constexpr type unpack(WideNum n) { return n.to<type>(dtype); }
  static constexpr WideNum pack(type x) {
    return WideNum::from<type>(dtype, x);
  }
  static constexpr WideNum widen(narrowtype unwide) {
    return pack(static_cast<type>(unwide));
  }
  static constexpr narrowtype narrow(WideNum wide) {
    return static_cast<narrowtype>(unpack(wide));
  }
};

} // namespace onnx_mlir