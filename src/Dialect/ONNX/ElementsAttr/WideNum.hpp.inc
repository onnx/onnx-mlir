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
  return wideZeroDispatch(type, [](auto wideZero) {
    using WideCppType = decltype(wideZero);
    return WideNumWrappedFunction<TemplateFunction<OP, WideCppType>>::eval;
  });
}

template <typename Res, typename Arg>
inline std::function<WideNum(WideNum)> widenumWrapped(
    std::function<Res(Arg)> lambda) {
  return [lambda = std::move(lambda)](detail::Packed<Arg> arg) -> WideNum {
    return detail::pack<Res>(lambda(detail::unpack<Arg>(arg)));
  };
}
