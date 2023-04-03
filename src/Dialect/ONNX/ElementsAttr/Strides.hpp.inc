//===-------------------------- Strides.hpp.inc ---------------------------===//
//
//===----------------------------------------------------------------------===//
// Strides template implementations
//===----------------------------------------------------------------------===//

template <typename Iterator, typename Arg0, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator dst,
    StridedArrayRef<Arg0> src0, Action &&act) {
  for (auto &ipos : StridesRange<1>(shape, {src0.strides})) {
    act(dst, src0.begin() + ipos.pos[0]);
    ++dst;
  }
  return dst;
}

template <typename Iterator, typename Arg0, typename Arg1, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator dst,
    StridedArrayRef<Arg0> src0, StridedArrayRef<Arg1> src1, Action &&act) {
  // The implementation below effectively inlines this iterator implementation
  // (manually inlined below to get 1.5x speedup):
  //
  //   for (auto &ipos : StridesRange<2>(shape, {src0.strides, src1.strides}))
  //     act(dst, src0.begin() + ipos.pos[0], src1.begin() + ipos.pos[1]);
  //   return dst;
  //
  if (llvm::any_of(shape, [](int64_t dimSize) { return dimSize == 0; })) {
    return dst; // No traversal because shape is empty.
  }
  size_t pos0 = 0, pos1 = 0;
  llvm::SmallVector<uint64_t, 6> index(shape.size(), 0);
  for (;;) {
    act(dst, src0.begin() + pos0, src1.begin() + pos1);
    ++dst;
    for (auto axis = shape.size();;) {
      if (axis == 0) {
        return dst; // Index rolled around.
      }
      --axis;
      uint64_t dim = shape[axis];
      pos0 += src0.strides[axis];
      pos1 += src1.strides[axis];
      if (++(index[axis]) < dim)
        break;
      pos0 -= dim * src0.strides[axis];
      pos1 -= dim * src1.strides[axis];
      index[axis] = 0;
    }
  }
  llvm_unreachable("traveral returns when index rolls around");
}

template <typename Res, typename Arg0, typename Action>
void mapStrides(llvm::ArrayRef<int64_t> shape, llvm::MutableArrayRef<Res> dst,
    StridedArrayRef<Arg0> src0, Action &&act) {
  Res *end = traverseStrides<Res *, Arg0>(shape, dst.begin(), src0,
      [&act](Res *res, const Arg0 *arg0) { *res = act(*arg0); });
  assert(end == dst.end() && "traverses every dst element");
}

template <typename Res, typename Arg0, typename Arg1, typename Action>
void mapStrides(llvm::ArrayRef<int64_t> shape, llvm::MutableArrayRef<Res> dst,
    StridedArrayRef<Arg0> src0, StridedArrayRef<Arg1> src1, Action &&act) {
  Res *end = traverseStrides<Res *, Arg0, Arg1>(shape, dst.begin(), src0, src1,
      [&act](Res *res, const Arg0 *arg0, const Arg1 *arg1) {
        *res = act(*arg0, *arg1);
      });
  assert(end == dst.end() && "traverses every dst element");
}
