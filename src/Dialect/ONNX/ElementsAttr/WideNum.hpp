/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- WideNum.hpp -----------------------------===//
//
// WideNum data type.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_WIDE_NUM_H
#define ONNX_MLIR_WIDE_NUM_H

#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"

#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace onnx_mlir {

// Union of 64-bit integers and double precision floating point numbers.
//
// It is tagless and should always be used in a conjunction with a BType,
// or a type t with implied BType btypeOfMlirType(t)
// or a shaped type s with implied BType btypeOfMlirType(s.getElementType()).
//
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

  // Converts dbl to an APFloat with the floating point semantics
  // corresponding to tag.
  // Precondition: tag must be a floating point type.
  llvm::APFloat toAPFloat(BType tag) const;

  // Returns WideNum with dbl set to the value of x.
  static WideNum fromAPFloat(llvm::APFloat x);

  // Converts i64 or u64, corresponding to the sign of tag, to an APInt with
  // bitwidth and sign corresponding to tag.
  // Precondition: tag must be a (signed or unsigned) integer type.
  llvm::APInt toAPInt(BType tag) const;

  // Returns WideNum with i64 or u64, corresponding to isSigned, set to the
  // value of x.
  static WideNum fromAPInt(llvm::APInt x, bool isSigned);

  template <typename T>
  constexpr T to(BType tag) const {
    switch (tag) {
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
    case BType::FLOAT8E4M3FN:
    case BType::FLOAT8E4M3FNUZ:
    case BType::FLOAT8E5M2:
    case BType::FLOAT8E5M2FNUZ:
      return static_cast<T>(dbl);
    default:
      llvm_unreachable("to unsupported btype");
    }
  }

  template <typename T>
  static constexpr WideNum from(BType tag, T x) {
    switch (tag) {
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
    case BType::FLOAT8E4M3FN:
    case BType::FLOAT8E4M3FNUZ:
    case BType::FLOAT8E5M2:
    case BType::FLOAT8E5M2FNUZ:
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

namespace detail {
// WideNumWrappedFunction helper.
template <typename FunctionType, class Function>
struct FunctionWrapper;
} // namespace detail

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
using WideNumWrappedFunction =
    detail::FunctionWrapper<decltype(Function::eval), Function>;

// If TemplateFunction<OP, T> is a Function class, like the argument to
// WrappedFunction, then getWideNumWrappedTemplateFunction instantiates it with
// the T corresponding to the given mlir type (promotes to the nearest among the
// 4 types double, int64_t, uint64_t, bool) and returns a function pointer to
// the instantiated static eval function, wrapped to take WideNum args and
// return WideNum result. See ConstProp.cpp for example uses.
//
// NOTE: Although we only pass two type arguments TemplateFunction is
//       declared with a variadic second argument typename... T
//       to support an extra 'Enable' type argument for enable_if stuff;
//       see ElementWiseBinaryOpImpl, ElementWiseUnaryOpImpl in ConstProp.cpp.
template <template <class OP, typename... T> class TemplateFunction, class OP>
auto getWideNumWrappedTemplateFunction(mlir::Type type);

// Returns a wrapped function with WideNum argument and return type.
// It unpacks the argument to the arument type Arg, calls the lambda, and
// takes the result of type Res and returns it packed as a WideNum.
// Types Res and Arg must be double, int64_t, uint64_t, or bool.
template <typename Res, typename Arg>
std::function<WideNum(WideNum)> widenumWrapped(std::function<Res(Arg)> lambda);

// Calls act with a zero of the C++ type corresponding to the given mlir type
// (promotes to the nearest among the 4 types double, int64_t, uint64_t, bool).
// The zero argument is a token which can be used to find the C++ type.
// Use it similarly to dispatchByBType: call it with a generic lambda which
// can read the C++ type of the argument with decltype. For instance, given
// a signed integer factor and a WideNum packed according to an mlir type
// you can multiply the WideNum with:
//
//   WideNum multiplyBy(int factor, WideNum n, Type type) {
//     return wideZeroDispatch(type, [factor, n](auto wideZero) {
//       constexpr BType TAG = toBType<decltype(wideZero)>;
//       return WideNum::widen<TAG>(factor * n.narrow<TAG>());
//     });
//   }
//
template <typename Action>
auto wideZeroDispatch(mlir::Type type, Action &&act);

template <typename Action>
auto wideZeroDispatchNonBool(mlir::Type type, Action &&act) {
  if (mlir::isa<mlir::FloatType>(type))
    return act(static_cast<double>(0));
  auto itype = mlir::cast<mlir::IntegerType>(type);
  if (itype.isUnsigned())
    return act(static_cast<uint64_t>(0));
  else
    return act(static_cast<int64_t>(0));
}

template <typename Action>
inline auto wideZeroDispatch(mlir::Type type, Action &&act) {
  if (type.isInteger(1))
    return act(static_cast<bool>(0));
  else
    return wideZeroDispatchNonBool(type, act);
}

// Include template implementations.
#include "WideNum.hpp.inc"

} // namespace onnx_mlir
#endif
