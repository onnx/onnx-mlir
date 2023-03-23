/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- Strides.hpp -----------------------------===//
//
// Strides helper functions.
//
// Strides are the same concept as in PyTorch and NumPy.
// A tensor's strides are an int64_t array with length == the tensor's rank
// and describe the layout of elements in a linear array. They can express
// things like row-major or column-major order, and they can express that the
// linear array is smaller than the tensor size and elements from the linear
// array should be broadcast to populate the tensor. In the extreme, the
// linear array can be a "splat" singleton broadcast to every tensor element.
//
// getStridesPosition(indices, strides) maps tensor indices to the position
// in the linear array by computing the dot product of indices and strides.
//
// A linear array and strides can represent a tensor with a given shape if
// getStridesPosition() maps the tensor indices onto the array positions,
// which happens when the last tensor indices map to the last array position:
//
//   getStridesPosition([shape[0]-1,...,shape[rank-1]-1], strides)
//   == array.size() - 1
//
// when shape is non-empty.
//
// Given a strided tensor (represented by a linear array and strides) it can
// always be transposed by just transposing the strides and can always be
// broadcast to a larger shape by just expanding the strides.
// On the other hand, reshaping a strided tensor sometimes requires reordering
// the elements in the linear array in contrast to, e.g.,
// DenseElementsAttr::reshape() which always reuses its linear array.
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/ElementsAttr/Arrays.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>

namespace onnx_mlir {

// Returns the position in the linear array described by the strides
// which correpond to the given indices.
size_t getStridesPosition(
    llvm::ArrayRef<int64_t> indices, llvm::ArrayRef<int64_t> strides);

// The data is splat (singleton) if strides are all zero.
inline bool areStridesSplat(llvm::ArrayRef<int64_t> strides) {
  return llvm::all_of(strides, [](int64_t s) { return s == 0; });
}

// Returns strides == getDefaultStrides(shape, strides).
bool areStridesContiguous(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides);

// Returns row-major order strides for the given shape.
llvm::SmallVector<int64_t, 4> getDefaultStrides(llvm::ArrayRef<int64_t> shape);

// Returns all-zeros strides.
llvm::SmallVector<int64_t, 4> getSplatStrides(llvm::ArrayRef<int64_t> shape);

// Returns the strides that can map the underlying data to reshapedShape
// equivalently to restriding it, if such strides exist, otherwise returns None.
llvm::Optional<llvm::SmallVector<int64_t, 4>> reshapeStrides(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
    llvm::ArrayRef<int64_t> reshapedShape);

// Returns strides that broadcast to the expandedShape under the assumption
// that the given strides represent a shape that broadcasts to expandedShape.
llvm::SmallVector<int64_t, 4> expandStrides(
    llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> expandedShape);

// The following transpose and unflatten functions are more about shapes than
// strides but they live here for now:

llvm::SmallVector<int64_t, 4> transposeDims(
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<uint64_t> perm);

llvm::SmallVector<int64_t, 4> untransposeDims(
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<uint64_t> perm);

// NOTE: this function is expensive, try to avoid calling it
llvm::SmallVector<int64_t, 4> unflattenIndex(
    llvm::ArrayRef<int64_t> shape, int64_t flatIndex);

// Unpacks src into row-major order in dstData.
void restrideArray(unsigned elementBytewidth, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> srcStrides, llvm::ArrayRef<char> src,
    llvm::MutableArrayRef<char> dst);

// Unpacks src into row-major order in dstData.
template <typename T>
void restrideArray(llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> srcStrides, llvm::ArrayRef<T> src,
    llvm::MutableArrayRef<T> dst) {
  // To reduce code size restrideArray() is only implemented once for each type
  // size so, e.g., int32_t, uint32_t, float all use the same implementation.
  return restrideArray(sizeof(T), shape, srcStrides, castArrayRef<char>(src),
      castMutableArrayRef<char>(dst));
}

// A linear array together with strides.
// Derives behavior including iterators from ArrayRef<T>.
// Tensor indices can be mapped to array positions with getStridesPosition().
template <typename T>
struct StridedArrayRef : public llvm::ArrayRef<T> {
  using Base = llvm::ArrayRef<T>;

  llvm::ArrayRef<int64_t> strides;
  StridedArrayRef(llvm::ArrayRef<T> array, llvm::ArrayRef<int64_t> strides)
      : Base(array), strides(strides) {}
};

// Suppose strided arrays {{array1,strides1},...,{arrayN,stridesN}} all
// represent the same shape, then
// it = StridesIterator<N>(shape, {strides1,...,stridesN}) iterates over the
// shape's elements in row-major order, with it->index describing the shape's
// corresponding multidimensional index, it->flattenedIndex the row-major order
// flattened linear index, and it->pos is an array of positions pos1,...,posN
// in array1,...,arrayN that represent the index. i.e.,
// it->pos1 == getStridesPosition(it->index, strides1) etc.
//
// For example, this can be used to compare the contents of two strided arrays
// with the same shape:
//
//   template <typename T, typename StridedArray = StridedArrayRef<T>>
//   bool equal(ArrayRef<int64_t> shape, StridedArray lhs, StridedArray rhs) {
//     StridesIterator<2> begin(shape, {lhs.strides, rhs.strides}), end(shape);
//     return llvm::all_of(llvm::make_range(begin, end),
//         [](const auto &it) { return src0[it.pos[0]] == src1[it.pos[1]]; })
//   }
//
// Note: The iteration example above works when shape is empty because
//       begin == end and the iterator is never derefenced or incremented.
//       If the shape is empty then the iterator shouldn't be dereferenced or
//       incremented because the internal state will not make sense.
//
template <size_t N>
class StridesIterator {
public:
  struct value_type {
    value_type(unsigned rank, uint64_t flattenedIndex = 0)
        : pos{}, flattenedIndex(flattenedIndex), index(rank, 0) {}
    std::array<size_t, N> pos;
    uint64_t flattenedIndex;
    llvm::SmallVector<uint64_t, 6> index;
  };
  using difference_type = int64_t;
  using pointer = const value_type *;
  using reference = const value_type &;
  using iterator_category = std::forward_iterator_tag;

private:
  const llvm::ArrayRef<int64_t> shape;
  const std::array<llvm::ArrayRef<int64_t>, N> strides;
  value_type value;

public:
  // Begin iterator.
  StridesIterator(llvm::ArrayRef<int64_t> shape,
      std::array<llvm::ArrayRef<int64_t>, N> strides)
      : shape(shape), strides(strides), value(shape.size()) {
    assert(!mlir::ShapedType::isDynamicShape(shape) && "shape must be static");
    for (unsigned i = 0; i < N; ++i)
      assert(shape.size() == strides[i].size() && "shape, strides mismatch");
  }

  // End iterator: ends after the given number of iterations.
  StridesIterator(size_t iterations) : value{0, iterations} {}

  // End iterator: ends after one traversal of shape.
  StridesIterator(llvm::ArrayRef<int64_t> shape)
      : StridesIterator(mlir::ShapedType::getNumElements(shape)) {}

  // These declarations are redundant, the compiler generates this constructor
  // and operator automatically, but included to flag that StridesIterator can
  // be copied, which is used in the implementation of operator++(int) below.
  StridesIterator(const StridesIterator &) = default;
  StridesIterator &operator=(const StridesIterator &) = default;

  bool operator==(const StridesIterator &other) const {
    return value.flattenedIndex == other.value.flattenedIndex;
  }

  bool operator!=(const StridesIterator &other) const {
    return value.flattenedIndex != other.value.flattenedIndex;
  }

  reference operator*() const { return value; }

  pointer operator->() const { return &value; }

  inline StridesIterator &operator++() {
    ++(value.flattenedIndex);
    for (auto axis = shape.size();;) {
      if (axis == 0)
        break;
      --axis;
      uint64_t dim = shape[axis];
      for (unsigned i = 0; i < N; ++i)
        value.pos[i] += strides[i][axis];
      if (++(value.index[axis]) < dim)
        break;
      // axis overflowed: rewind the axis and carry over to axis-1 by doing
      // the next iteration of the loop
      for (unsigned i = 0; i < N; ++i)
        value.pos[i] -= dim * strides[i][axis];
      value.index[axis] = 0;
    }
    return *this;
  }

  StridesIterator operator++(int) {
    StridesIterator copy = *this;
    ++*this;
    return copy;
  }
};

template <size_t N>
inline auto makeStridesIteratorRange(llvm::ArrayRef<int64_t> shape,
    std::array<llvm::ArrayRef<int64_t>, N> strides) {
  return llvm::make_range(
      StridesIterator<N>(shape, strides), StridesIterator<N>(shape));
}

template <typename Iterator, typename Arg0,
    typename Action = llvm::function_ref<void(Iterator, const Arg0 *)>>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator dst,
    StridedArrayRef<Arg0> src0, Action &&act);

template <typename Iterator, typename Arg0, typename Arg1,
    typename Action =
        llvm::function_ref<void(Iterator, const Arg0 *, const Arg1 *)>>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator dst,
    StridedArrayRef<Arg0> src0, StridedArrayRef<Arg1> src1, Action &&act);

template <typename Res, typename Arg0,
    typename Action = llvm::function_ref<Res(Arg0)>>
void mapStrides(llvm::ArrayRef<int64_t> shape, llvm::MutableArrayRef<Res> dst,
    StridedArrayRef<Arg0> src0, Action &&act);

template <typename Res, typename Arg0, typename Arg1,
    typename Action = llvm::function_ref<Res(Arg0, Arg1)>>
void mapStrides(llvm::ArrayRef<int64_t> shape, llvm::MutableArrayRef<Res> dst,
    StridedArrayRef<Arg0> src0, StridedArrayRef<Arg1> src1, Action &&act);

// Include template implementations.
#include "Strides.hpp.inc"

} // namespace onnx_mlir