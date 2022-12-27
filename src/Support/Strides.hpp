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

#include "src/Support/Arrays.hpp"
#include "src/Support/WideNum.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

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

// A linear array together with strides.
// Tensor indices can be mapped to array positions with getStridesPosition().
template <typename T>
struct Strided {
  llvm::ArrayRef<int64_t> strides;
  T data;
};

// Unpacks src into row-major order in dstData.
void restrideArray(unsigned elementBytewidth, llvm::ArrayRef<int64_t> shape,
    Strided<llvm::ArrayRef<char>> src, llvm::MutableArrayRef<char> dstData);

// Unpacks src into row-major order in dstData.
template <typename T>
void restrideArray(llvm::ArrayRef<int64_t> shape,
    Strided<llvm::ArrayRef<T>> src, llvm::MutableArrayRef<T> dstData) {
  return restrideArray(sizeof(T), shape,
      {src.strides, castArrayRef<char>(src.data)},
      castMutableArrayRef<char>(dstData));
}

template <typename BinaryFunction = std::function<WideNum(WideNum, WideNum)>>
inline void transformAndRestrideTwoWideArrays(llvm::ArrayRef<int64_t> shape,
    Strided<llvm::ArrayRef<WideNum>> lhs, Strided<llvm::ArrayRef<WideNum>> rhs,
    llvm::MutableArrayRef<WideNum> dstData, BinaryFunction fun) {
  assert(lhs.strides.size() == shape.size() && "lhs strides must be expanded");
  assert(rhs.strides.size() == shape.size() && "rhs strides must be expanded");
  size_t rank = shape.size();
  auto traverse = [=](size_t axis, size_t lhsPos, size_t rhsPos, WideNum *dst,
                      const auto &recurse) -> WideNum * {
    if (axis == rank) {
      *dst = fun(lhs.data[lhsPos], rhs.data[rhsPos]);
      return dst + 1;
    } else {
      size_t lhsStride = lhs.strides[axis];
      size_t rhsStride = rhs.strides[axis];
      size_t dimSize = shape[axis];
      for (size_t i = 0; i < dimSize; ++i) {
        dst = recurse(axis + 1, lhsPos, rhsPos, dst, recurse);
        lhsPos += lhsStride;
        rhsPos += rhsStride;
      }
      return dst;
    }
  };
  WideNum *end = traverse(0, 0, 0, dstData.begin(), traverse);
  assert(end == dstData.end() && "traversal should end at the end of dstData");
}

} // namespace onnx_mlir