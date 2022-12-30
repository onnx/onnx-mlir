//===-------------------------- Strides.hpp.inc ---------------------------===//
//
//===----------------------------------------------------------------------===//
// Strides template implementations
//===----------------------------------------------------------------------===//

namespace detail {
template <size_t N, typename Action>
struct StridesTraversal {
  const llvm::ArrayRef<int64_t> shape;
  const std::array<llvm::ArrayRef<int64_t>, N> &strides;
  Action &&act;
  StridesTraversal(llvm::ArrayRef<int64_t> shape,
      const std::array<llvm::ArrayRef<int64_t>, N> &strides, Action &&act)
      : shape(shape), strides(strides), act(std::forward<Action>(act)) {}
  size_t traverse() const { return doTraverse(0, 0, {}); }
  size_t doTraverse(
      unsigned axis, uint64_t index, std::array<uint64_t, N> positions) const {
    unsigned rank = shape.size();
    if (axis == rank) {
      act(index, positions);
      index++;
    } else {
      for (int64_t i = 0; i < shape[axis]; ++i) {
        index = doTraverse(axis + 1, index, positions);
        for (size_t j = 0; j < N; ++j)
          positions[j] += strides[j][axis];
      }
    }
    return index;
  }
};
} // namespace detail

template <size_t N, typename Action>
void traverseStrides(llvm::ArrayRef<int64_t> shape,
    const std::array<llvm::ArrayRef<int64_t>, N> &strides, Action &&act) {
  detail::StridesTraversal<2, Action>(shape, strides, std::forward<Action>(act))
      .traverse();
}
