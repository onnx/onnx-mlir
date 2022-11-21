/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- WideNum.hpp -----------------------------===//
//
// WideNum data type.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Support/BType.hpp"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"

namespace onnx_mlir {

// Union of 64-bit integers and double precision floating point numbers.
// It is tagless and should always be used in a conjunction with a BType.
// The btype tags which field of the union is populated:
// dbl if isFloat(btype), i64 or u64 if isSigned/UnsignedInt(btype).
//
// WideNum satisfies for all cpp types X and Y values x of type X
//
//   static_cast<Y>(x) == WideNum::from<X>(btype, x).To<Y>(btype)
//
// provided the wide type of btype (double, int64_t, uint64_t) has enough
// precision and range to represent x up to the precision and range of Y.
//
union WideNum {
  double dbl;   // Floating point numbers with precision and range up to double.
  int64_t i64;  // Signed ints up to bitwidth 64.
  uint64_t u64; // Unsigned ints up to bitwidth 64, including bool.

  llvm::APFloat toAPFloat(BType tag) const;

  static WideNum fromAPFloat(BType tag, llvm::APFloat x);

  llvm::APInt toAPInt(BType tag) const;

  static WideNum fromAPInt(BType tag, llvm::APInt x);

  template <typename T>
  constexpr T to(BType dtag) const {
    switch (dtag) {
    case BType::BOOL:
    case BType::UINT8:
    case BType::UINT16:
    case BType::UINT32:
    case BType::UINT64:
      return static_cast<T>(u64);
    case BType::INT8:
    case BType::INT16:
    case BType::INT32:
    case BType::INT64:
      return static_cast<T>(i64);
    case BType::DOUBLE:
    case BType::FLOAT:
    case BType::FLOAT16:
    case BType::BFLOAT16:
      return static_cast<T>(dbl);
    default:
      llvm_unreachable("to unsupported btype");
    }
  }

  template <typename T>
  static constexpr WideNum from(BType dtag, T x) {
    switch (dtag) {
    case BType::BOOL:
    case BType::UINT8:
    case BType::UINT16:
    case BType::UINT32:
    case BType::UINT64:
      return {.u64 = static_cast<uint64_t>(x)};
    case BType::INT8:
    case BType::INT16:
    case BType::INT32:
    case BType::INT64:
      return {.i64 = static_cast<int64_t>(x)};
    case BType::DOUBLE:
    case BType::FLOAT:
    case BType::FLOAT16:
    case BType::BFLOAT16:
      return {.dbl = static_cast<double>(x)};
    default:
      llvm_unreachable("from unsupported btype");
    }
  }

  void store(BType dtag, llvm::MutableArrayRef<char> memory) const;

  static WideNum load(BType dtag, llvm::ArrayRef<char> memory);
};
static_assert(sizeof(WideNum) * CHAR_BIT == 64, "WideNum is 64 bits wide");

template <BType DTYPE>
struct WideBType {
  using narrowtype = CppType<DTYPE>;
  using type = typename BTypeTrait<DTYPE>::widetype;
  static constexpr BType btype = toBType<type>;
  static constexpr type unpack(WideNum n) { return n.to<type>(btype); }
  static constexpr WideNum pack(type x) {
    return WideNum::from<type>(btype, x);
  }
  static constexpr WideNum widen(narrowtype unwide) {
    return pack(static_cast<type>(unwide));
  }
  static constexpr narrowtype narrow(WideNum wide) {
    return static_cast<narrowtype>(unpack(wide));
  }
};

} // namespace onnx_mlir