/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- Strides.hpp -----------------------------===//
//
// Strides helper functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Support/WideNum.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace onnx_mlir {

int64_t getStridesNumElements(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides);

size_t getStridesPosition(
    llvm::ArrayRef<int64_t> indices, llvm::ArrayRef<int64_t> strides);

// The data is splat (singleton) if strides are all zero.
inline bool areStridesSplat(llvm::ArrayRef<int64_t> strides) {
  return llvm::all_of(strides, [](int64_t s) { return s == 0; });
}

// Returns strides == getDefaultStrides(shape, strides).
bool areStridesContiguous(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides);

llvm::SmallVector<int64_t, 4> getDefaultStrides(llvm::ArrayRef<int64_t> shape);

llvm::Optional<llvm::SmallVector<int64_t, 4>> transposeStrides(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
    llvm::ArrayRef<uint64_t> perm);

llvm::Optional<llvm::SmallVector<int64_t, 4>> reshapeStrides(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
    llvm::ArrayRef<int64_t> reshapedShape);

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

template <typename T>
struct Strided {
  llvm::ArrayRef<int64_t> strides;
  T data;
};

void restrideArray(unsigned elementBytewidth, llvm::ArrayRef<int64_t> shape,
    Strided<llvm::ArrayRef<char>> src,
    Strided<llvm::MutableArrayRef<char>> dst);

// Computes dstStrides from shape, and pads them and srcStrides.
inline void restrideArray(unsigned elementBytewidth,
    llvm::ArrayRef<int64_t> shape, Strided<llvm::ArrayRef<char>> src,
    llvm::MutableArrayRef<char> dstData) {
  auto dstStrides = getDefaultStrides(shape);
  Strided<llvm::MutableArrayRef<char>> dst{dstStrides, dstData};
  return restrideArray(elementBytewidth, shape, src, dst);
}

template <typename BinaryFunction = std::function<WideNum(WideNum, WideNum)>>
inline void transformAndRestrideTwoWideArrays(llvm::ArrayRef<int64_t> shape,
    Strided<llvm::ArrayRef<WideNum>> lhs, Strided<llvm::ArrayRef<WideNum>> rhs,
    Strided<llvm::MutableArrayRef<WideNum>> dst, BinaryFunction fun) {
  assert(lhs.strides.size() == shape.size() && "lhs strides must be padded");
  assert(rhs.strides.size() == shape.size() && "rhs strides must be padded");
  assert(dst.strides.size() == shape.size() && "dst strides must be full rank");
  int64_t rank = shape.size();
  auto traverse = [=](int64_t axis, size_t lhsPos, size_t rhsPos, size_t dstPos,
                      const auto &recurse) -> void {
    if (axis == rank) {
      dst.data[dstPos] = fun(lhs.data[lhsPos], rhs.data[rhsPos]);
    } else {
      size_t lhsStride = lhs.strides[axis];
      size_t rhsStride = rhs.strides[axis];
      size_t dstStride = dst.strides[axis];
      size_t dimSize = shape[axis];
      for (size_t i = 0; i < dimSize; ++i) {
        recurse(axis + 1, lhsPos, rhsPos, dstPos, recurse);
        lhsPos += lhsStride;
        rhsPos += rhsStride;
        dstPos += dstStride;
      }
    }
  };
  traverse(0, 0, 0, 0, traverse);
}

} // namespace onnx_mlir