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
// getStridesPosition(index, strides) maps a multidimensional tensor index to
// the linear array position given by the dot product of index and strides.
//
// A linear array and strides can represent a tensor with a given shape if
// getStridesPosition() maps the tensor indices onto the array positions,
// which happens when the last tensor index map to the last array position:
//
//   getStridesPosition({[shape[0]-1,...,shape[rank-1]-1]}, strides)
//   == array.size() - 1
//
// provided shape is non-empty.
//
// Given a strided tensor (represented by a linear array and strides) it can
// always be transposed by just transposing the strides and can always be
// broadcast to a larger shape by just expanding the strides.
// On the other hand, reshaping a strided tensor sometimes requires reordering
// the elements in the linear array in contrast to, e.g.,
// DenseElementsAttr::reshape() which always reuses its linear array.
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_STRIDES_H
#define ONNX_MLIR_STRIDES_H

#include "src/Support/Arrays.hpp"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace onnx_mlir {

// Returns the position in the linear array described by the strides
// which correpond to the given index.
//
// Note that the uint64_t type and the name 'index' for the multidimensional
// index, as well as the name 'flattenedIndex' in unflattenIndex() below, are
// borrowed from the mlir::ElementsAttr attribute interface methods.
uint64_t getStridesPosition(
    llvm::ArrayRef<uint64_t> index, llvm::ArrayRef<int64_t> strides);

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
std::optional<llvm::SmallVector<int64_t, 4>> reshapeStrides(
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
llvm::SmallVector<uint64_t, 4> unflattenIndex(
    llvm::ArrayRef<int64_t> shape, uint64_t flattenedIndex);

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

} // namespace onnx_mlir
#endif