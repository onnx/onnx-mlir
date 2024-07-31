/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- StridesRange.hpp --------------------------===//
//
// StridesRange class for iterating over strided tensors.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_STRIDES_RANGE_H
#define ONNX_MLIR_STRIDES_RANGE_H

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>

namespace onnx_mlir {

// Used as value_type in StridesRange and StridesIterator in the context of a
// shape of the given rank (the length of the 'index' vector) and N strides
// which are not represented in StridesIndexOffsets itself.
//
// 'flattenedIndex' and 'index' are two representations of an index into a
// tensor of the given rank.
// 'offsets' is an array of flattened offsets into strided arrays. Can be
// negative if strides are negative, e.g. to represent negative slice steps.
template <size_t N>
struct StridesIndexOffsets {
  // Non-zero flattenedIndex is only used to construct end iterators
  // with meaningless index and offsets.
  StridesIndexOffsets(unsigned rank, uint64_t flattenedIndex = 0)
      : flattenedIndex(flattenedIndex), index(rank, 0), offsets{} {}
  uint64_t flattenedIndex;
  llvm::SmallVector<uint64_t, 6> index;
  std::array<int64_t, N> offsets;
  int64_t at(size_t i) const {
    assert(i < N && "index out of range");
    return offsets[i];
  }
  int64_t operator[](size_t i) const { return at(i); }
};

// Suppose (array0,strides0),...,(arrayN,stridesN) are all strided tensors with
// the same shape, then
// it = StridesIterator<N+1>(shape, {strides0,...,stridesN}) iterates over the
// shape's elements in row-major order, with it->index describing the shape's
// corresponding multidimensional index, it->flattenedIndex the row-major order
// flattened linear index, and it->offsets is an array of flat linear indexes
// offset0,...,offsetN in array0,...,arrayN that represent the index, i.e.,
// it->at(0) == it->offsets[0] == getStridesPosition(it->index, strides0) etc.
//
// For example, this can be used to compare the contents of two strided tensors
// (a0,strides0) and (a1,strides1) with the same shape:
//
//   template <typename T, typename A = ArrayRef<T>, S = ArrayRef<int64_t>>
//   bool equal(S shape, A a0, S strides0, A a1, S strides1) {
//     StridesIterator<2> begin(shape, {strides0, strides1}), end(shape);
//     return std::all_of(begin, end,
//       [](const auto &idxoffs) { return a0[idxoffs[0]] == a1[idxoffs[1]]; });
//   }
//
// Note: The iteration example above works when shape is empty because
//       begin == end and the iterator is never derefenced or incremented.
//       If the shape is empty then the iterator shouldn't be dereferenced or
//       incremented because the internal state will not make sense.
//
// It is best to access StridesIterator through the StridesRange wrapper below.
template <size_t N>
class StridesIterator {
public:
  using value_type = StridesIndexOffsets<N>;
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
  StridesIterator(size_t iterations) : strides{}, value{0, iterations} {}

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

  StridesIterator &operator++();

  StridesIterator operator++(int) {
    StridesIterator copy = *this;
    ++*this;
    return copy;
  }
};

// StridesRange<N>(shape, strides) is almost the same as
// make_range(StridesIterator<N>(shape, strides), StridesIterator<N>(shape))
// but a little more concise and efficient to use.
//
// For example, comparison of two strided tensors with the same shape:
//
//   template <typename T, typename A = ArrayRef<T>, S = ArrayRef<int64_t>>
//   bool equal(S shape, A a0, S strides0, A a1, S strides1) {
//     return llvm::all_of(StridesRange<2>(shape, {strides0, strides1}),
//       [](const auto &idxoffs) { return a0[idxoffs[0]] == a1[idxoffs[1]]; });
//   }
template <size_t N>
class StridesRange {
public:
  using iterator = StridesIterator<N>;
  using value_type = StridesIndexOffsets<N>;

private:
  const llvm::ArrayRef<int64_t> shape;
  const std::array<llvm::ArrayRef<int64_t>, N> strides;
  const size_t numElements;

public:
  StridesRange(llvm::ArrayRef<int64_t> shape,
      std::array<llvm::ArrayRef<int64_t>, N> strides)
      : shape(shape), strides(strides),
        numElements(mlir::ShapedType::getNumElements(shape)) {}
  iterator begin() const { return iterator(shape, strides); }
  iterator end() const { return iterator(numElements); }
  size_t size() const { return numElements; }
  bool empty() const { return size() == 0; }
};

// Include template implementations.
#include "StridesRange.hpp.inc"

} // namespace onnx_mlir
#endif