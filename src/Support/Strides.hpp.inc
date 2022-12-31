//===-------------------------- Strides.hpp.inc ---------------------------===//
//
//===----------------------------------------------------------------------===//
// Strides template implementations
//===----------------------------------------------------------------------===//

template <typename Iterator, typename... Args, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator begin,
    StridedArrayRef<Args>... src, Action &&act) {
  auto traverse = [=](const auto &recurse, unsigned axis, Iterator it,
                      const Args *... elms) -> Iterator {
    if (axis == shape.size()) {
      act(it, *elms...);
      it++;
    } else {
      for (int64_t i = 0; i < shape[axis]; ++i) {
        it = recurse(recurse, axis + 1, it, elms...);
        ((elms += src.strides[axis]), ...);
      }
    }
    return it;
  };
  return traverse(traverse, /*axis=*/0, begin, src.begin()...);
}

template <typename Res, typename... Args, typename Action>
void mapStrides(llvm::ArrayRef<int64_t> shape, llvm::MutableArrayRef<Res> dst,
    StridedArrayRef<Args>... src, Action &&act) {
  Res *end = traverseStrides<Res *, Args...>(shape, dst.begin(), src...,
      [&act](Res *res, Args... args) { *res = act(args...); });
  assert(end == dst.end() && "traverses every dst element");
}
