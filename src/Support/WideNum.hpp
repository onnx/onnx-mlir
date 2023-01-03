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

#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
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
      return WideNum(static_cast<uint64_t>(x)); // .u64
    case BType::INT8:
    case BType::INT16:
    case BType::INT32:
    case BType::INT64:
      return WideNum(static_cast<int64_t>(x)); // .i64
    case BType::DOUBLE:
    case BType::FLOAT:
    case BType::FLOAT16:
    case BType::BFLOAT16:
      return WideNum(static_cast<double>(x)); // .dbl
    default:
      llvm_unreachable("from unsupported btype");
    }
  }

  // "Narrows" the WideNum to the TAG's cpp type.
  template <BType TAG>
  constexpr CppType<TAG> narrow() const {
    return to<CppType<TAG>>(TAG);
  }

  // "Widens" (or "promotes") a value of the TAG's cpp type to a WideNum.
  template <BType TAG>
  static constexpr WideNum widen(CppType<TAG> x) {
    return from<CppType<TAG>>(TAG, x);
  }

  // WrappedFunction helper.
  template <typename FunctionType, class Function>
  struct FunctionWrapper;

  // Given a class Function with a static method
  //
  //   static ResType eval(ArgType...)
  //
  // where every ResType and ArgType is one of double, int64_t, uint64_t, bool,
  // constructs a class WrappedFunction<Function> with a static method
  //
  //   static WideNum eval(WideNum...)
  //
  // which unpacks the args, calls Function::eval, and packs the result.
  template <class Function>
  using WrappedFunction = FunctionWrapper<decltype(Function::eval), Function>;

  // If FunctionTemplate<OP, T> is a Function class, like the argument to
  // WrappedFunction, then wrappedTemplateFunction instantiates it with the T
  // corresponding to the given mlir type (promotes to the nearest among the
  // 4 types double, int64_t, uint64_t, bool) and returns a function pointer to
  // the instantiated static eval function, wrapped to take WideNum args and
  // return WideNum result. See ConstProp.cpp for example uses.
  //
  // NOTE: Although we only pass two type arguments to FunctionTemplate is
  //       declared with a variadic second argument typename... T
  //       to support an extra 'Enable' type argument for enable_if stuff;
  //       see ElementWiseBinaryOpImpl, ElementWiseUnaryOpImpl in ConstProp.cpp.
  template <template <class OP, typename... T> class FunctionTemplate, class OP>
  static inline constexpr auto wrappedTemplateFunction(mlir::Type type) {
    if (auto itype = type.dyn_cast<mlir::IntegerType>()) {
      if (itype.getWidth() == 1) {
        return WrappedFunction<FunctionTemplate<OP, bool>>::eval;
      } else if (itype.isUnsigned()) {
        return WrappedFunction<FunctionTemplate<OP, uint64_t>>::eval;
      } else {
        return WrappedFunction<FunctionTemplate<OP, int64_t>>::eval;
      }
    } else {
      assert(type.isa<mlir::FloatType>());
      return WrappedFunction<FunctionTemplate<OP, double>>::eval;
    }
  }

private:
  template <typename X>
  static inline constexpr bool isPackable =
      llvm::is_one_of<X, double, int64_t, uint64_t, bool>::value;

  // unpack<X>(n) is like reinterpret_cast<X>(n).
  template <typename X>
  constexpr static X unpack(WideNum n) {
    assert(isPackable<X>);
    return n.narrow<toBType<X>>(); // == n.to<X>(toBType<X>);
  }

  // pack<X>(x) is like reinterpret_cast<WideNum>(x).
  template <typename X>
  static constexpr WideNum pack(X x) {
    assert(isPackable<X>);
    return widen<toBType<X>>(x); // == from<X>(toBType<X>, x);
  }

  template <class Function, typename Res, typename... Args>
  struct FunctionWrapper<Res(Args...), Function> {
    template <typename T>
    using Packed = WideNum;

    static WideNum eval(Packed<Args>... args) {
      return WideNum::pack<Res>(Function::eval(unpack<Args>(args)...));
    }
  };

private:
  // TODO: With C++20 eliminate these constructors and replace all uses
  //       with designated initializers {.u64 = ..}, {.i64 = ..}, etc.
  //       and the default constructors and assignment below become implicit.
  constexpr explicit WideNum(uint64_t u64) : u64(u64) {}
  constexpr explicit WideNum(int64_t i64) : i64(i64) {}
  constexpr explicit WideNum(double dbl) : dbl(dbl) {}

public:
  WideNum() = default;
  constexpr WideNum(const WideNum &) = default;
  WideNum &operator=(const WideNum &) = default;
};
static_assert(sizeof(WideNum) * CHAR_BIT == 64, "WideNum is 64 bits wide");

} // namespace onnx_mlir
