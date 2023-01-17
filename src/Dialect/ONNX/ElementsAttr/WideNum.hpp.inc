//===-------------------------- WideNum.hpp.inc ---------------------------===//
//
//===----------------------------------------------------------------------===//
// WideNum template implementations
//===----------------------------------------------------------------------===//

namespace detail {

template <typename X>
inline constexpr bool isWideCppType =
    llvm::is_one_of<std::decay_t<X>, double, int64_t, uint64_t, bool>::value;

// unpack<X>(n) is like reinterpret_cast<X>(n).
template <typename X>
constexpr X unpack(WideNum n) {
  static_assert(isWideCppType<X>);
  return n.narrow<toBType<X>>(); // == n.to<X>(toBType<X>);
}

// pack<X>(x) is like reinterpret_cast<WideNum>(x).
template <typename X>
constexpr WideNum pack(X x) {
  static_assert(isWideCppType<X>);
  return WideNum::widen<toBType<X>>(x); // == from<X>(toBType<X>, x);
}

// Packed<Args>... is a WideNum sequence of the same length as Args.
template <typename T>
using Packed = WideNum;

template <class Function, typename Res, typename... Args>
struct FunctionWrapper<Res(Args...), Function> {
  static WideNum eval(Packed<Args>... args) {
    return pack<Res>(Function::eval(unpack<Args>(args)...));
  }
};

} // namespace detail

template <template <class OP, typename... T> class TemplateFunction, class OP>
inline auto getWideNumWrappedTemplateFunction(mlir::Type type) {
  if (auto itype = type.dyn_cast<mlir::IntegerType>()) {
    if (itype.getWidth() == 1) {
      return WideNumWrappedFunction<TemplateFunction<OP, bool>>::eval;
    } else if (itype.isUnsigned()) {
      return WideNumWrappedFunction<TemplateFunction<OP, uint64_t>>::eval;
    } else {
      return WideNumWrappedFunction<TemplateFunction<OP, int64_t>>::eval;
    }
  } else {
    assert(type.isa<mlir::FloatType>());
    return WideNumWrappedFunction<TemplateFunction<OP, double>>::eval;
  }
}