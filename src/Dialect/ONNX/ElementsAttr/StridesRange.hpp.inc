//===------------------------ StridesRange.hpp.inc ------------------------===//
//
//===----------------------------------------------------------------------===//
// StridesRange template implementations
//===----------------------------------------------------------------------===//

template <size_t N>
inline auto StridesIterator<N>::operator++() -> StridesIterator<N> & {
  ++(value.flattenedIndex);
  for (size_t axis = shape.size();;) {
    if (axis == 0) {
      break; // index rolled around
    }
    --axis;
    uint64_t dim = shape[axis];
    for (unsigned i = 0; i < N; ++i)
      value.offsets[i] += strides[i][axis];
    if (++(value.index[axis]) < dim)
      break;
    // axis overflowed: rewind the axis and carry over to axis-1 by doing
    // the next iteration of the loop
    for (unsigned i = 0; i < N; ++i)
      value.offsets[i] -= dim * strides[i][axis];
    value.index[axis] = 0;
  }
  return *this;
}

// This specialization for N=2 has been measured to speed up the
// ElementsAttrBuilder::combine() inner loop leading to a 20% speed up of
// ConstPropElementwiseBinary on the Apple M1 Pro chip with Apple clang 14.0.3.
// Ideally, it can be removed in the future once c++ compilers generate as
// efficient code from the unspecialized implementation above.
template <>
inline auto StridesIterator<2>::operator++() -> StridesIterator<2> & {
  ++(value.flattenedIndex);
  uint64_t *index = value.index.data();
  int64_t offset0 = value.offsets[0];
  int64_t offset1 = value.offsets[1];
  const int64_t *strides0 = strides[0].data();
  const int64_t *strides1 = strides[1].data();
  for (size_t axis = shape.size();;) {
    if (axis == 0) {
      break; // index rolled around
    }
    --axis;
    uint64_t dim = shape[axis];
    offset0 += strides0[axis];
    offset1 += strides1[axis];
    if (++(index[axis]) < dim)
      break;
    // axis overflowed: rewind the axis and carry over to axis-1 by doing
    // the next iteration of the loop
    offset0 -= dim * strides0[axis];
    offset1 -= dim * strides1[axis];
    index[axis] = 0;
  }
  value.offsets[0] = offset0;
  value.offsets[1] = offset1;
  return *this;
}
