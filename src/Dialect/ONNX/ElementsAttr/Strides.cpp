/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- Strides.cpp -----------------------------===//
//
// Strides helper functions.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/Strides.hpp"

#include "src/Dialect/ONNX/ElementsAttr/StridesRange.hpp"
#include "src/Support/Arrays.hpp"

using namespace mlir;

namespace onnx_mlir {

uint64_t getStridesPosition(
    ArrayRef<uint64_t> index, ArrayRef<int64_t> strides) {
  // Assert is commented out because this function is "on the fast path" called
  // for every element when iterating over DisposableElementsAttr values.
  // assert(index.size() == strides.size());
  uint64_t pos = 0;
  for (size_t axis = 0; axis < index.size(); ++axis)
    pos += index[axis] * strides[axis];
  return pos;
}

bool areStridesContiguous(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
  unsigned rank = shape.size();
  assert(rank == strides.size());
  int64_t mult = 1;
  for (int axis = rank - 1; axis >= 0; --axis) {
    int64_t dimSize = shape[axis];
    if (strides[axis] != (dimSize == 1 ? 0 : mult))
      return false;
    mult *= dimSize;
  }
  return true;
}

SmallVector<int64_t, 4> getDefaultStrides(ArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  SmallVector<int64_t, 4> strides;
  strides.resize_for_overwrite(rank);
  int64_t mult = 1;
  for (int64_t axis = rank - 1; axis >= 0; --axis) {
    int64_t dimSize = shape[axis];
    strides[axis] = dimSize == 1 ? 0 : mult;
    mult *= dimSize;
  }
  return strides;
}

SmallVector<int64_t, 4> getSplatStrides(ArrayRef<int64_t> shape) {
  return SmallVector<int64_t, 4>(shape.size(), 0);
}

std::optional<SmallVector<int64_t, 4>> reshapeStrides(ArrayRef<int64_t> shape,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> reshapedShape) {
  assert(shape.size() == strides.size());
  assert(ShapedType::getNumElements(shape) ==
         ShapedType::getNumElements(reshapedShape));

  if (areStridesContiguous(shape, strides))
    return getDefaultStrides(reshapedShape);

  assert(ShapedType::getNumElements(shape) > 1 &&
         "sizes < 2 are always contiguous");

  size_t rank1 = shape.size(), rank2 = reshapedShape.size();
  size_t a1 = 0, a2 = 0;
  SmallVector<int64_t, 4> reshapedStrides;
  do {
    assert(a2 == reshapedStrides.size());

    // Multiply dimSizes of leading axes with zero strides.
    int64_t m = 1;
    while (a1 < rank1 && strides[a1] == 0) {
      m *= shape[a1];
      ++a1;
    }
    // Add zero strides for axes in reshapedShape with dimSizes product m.
    int64_t m2 = 1;
    while (a2 < rank2 && m2 * reshapedShape[a2] <= m) {
      m2 *= reshapedShape[a2];
      reshapedStrides.push_back(0);
      ++a2;
    }
    if (m2 < m)
      return std::nullopt;
    if (a1 == rank1)
      break;

    assert(a2 == reshapedStrides.size());

    // Multiply dimSizes of contiguous leading axes. See:
    // https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
    assert(shape[a1] > 1);
    assert(strides[a1] > 0);
    int64_t n = 1;
    int64_t total = strides[a1] * shape[a1];
    int64_t last;
    do {
      n *= shape[a1];
      last = strides[a1];
      ++a1;
      while (a1 < rank1 && shape[a1] == 1) { // Skip dimSize 1 axes.
        assert(strides[a1] == 0);
        ++a1;
      }
    } while (a1 < rank1 && last == shape[a1] * strides[a1]);
    assert(total == n * last);
    // Add contiguous strides for axes in reshapedShape with dimSizes product n.
    int64_t n2 = 1;
    while (a2 < rank2 && n2 * reshapedShape[a2] <= n) {
      if (reshapedShape[a2] == 1) {
        reshapedStrides.push_back(0);
      } else {
        n2 *= reshapedShape[a2];
        total /= reshapedShape[a2];
        reshapedStrides.push_back(total);
      }
      ++a2;
    }
    if (n2 < n)
      return std::nullopt;
    assert(last == total);
  } while (a1 < rank1);
  assert(a2 == rank2);
  assert(a2 == reshapedStrides.size());
  return reshapedStrides;
}

SmallVector<int64_t, 4> expandStrides(
    ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> expandedShape) {
  size_t rank = expandedShape.size();
  assert(rank >= strides.size());
  SmallVector<int64_t, 4> padded(rank - strides.size(), 0);
  padded.append(strides.begin(), strides.end());
  return padded;
}

SmallVector<int64_t, 4> transposeDims(
    ArrayRef<int64_t> dims, ArrayRef<uint64_t> perm) {
  assert(dims.size() == perm.size());
  SmallVector<int64_t, 4> permutedDims;
  permutedDims.reserve(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    permutedDims.push_back(dims[perm[i]]);
  return permutedDims;
}

SmallVector<int64_t, 4> untransposeDims(
    ArrayRef<int64_t> dims, ArrayRef<uint64_t> perm) {
  assert(dims.size() == perm.size());
  SmallVector<int64_t, 4> unpermutedDims;
  unpermutedDims.resize_for_overwrite(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    unpermutedDims[perm[i]] = dims[i];
  return unpermutedDims;
}

SmallVector<uint64_t, 4> unflattenIndex(
    ArrayRef<int64_t> shape, uint64_t flattenedIndex) {
  SmallVector<uint64_t, 4> index;
  size_t rank = shape.size();
  if (rank > 0) {
    index.resize_for_overwrite(rank);
    for (size_t axis = rank - 1; axis >= 1; --axis) {
      assert(shape[axis] > 0 && "cannot unflatten shape with zeros");
      uint64_t dimSize = shape[axis];
      uint64_t rem = flattenedIndex % dimSize;
      flattenedIndex /= dimSize;
      index[axis] = rem;
    }
    assert(static_cast<int64_t>(flattenedIndex) < shape[0]);
    index[0] = flattenedIndex;
  }
  return index;
}

namespace {
template <typename T>
void restrideArrayImpl(unsigned elementBytewidth, ArrayRef<int64_t> shape,
    ArrayRef<int64_t> srcStrides, ArrayRef<char> src,
    MutableArrayRef<char> dst) {
  assert(sizeof(T) == elementBytewidth && "dispatch safety check");
  ArrayRef<T> srcT = castArrayRef<T>(src);
  MutableArrayRef<T> dstT = castMutableArrayRef<T>(dst);
  for (auto &idxoffs : StridesRange<1>(shape, {srcStrides}))
    dstT[idxoffs.flattenedIndex] = srcT[idxoffs[0]];
}
} // namespace

void restrideArray(unsigned elementBytewidth, ArrayRef<int64_t> shape,
    ArrayRef<int64_t> srcStrides, ArrayRef<char> src,
    MutableArrayRef<char> dst) {
  auto xpSrcStrides = expandStrides(srcStrides, shape);
  // clang-format off
  switch (elementBytewidth) {
  case 1: return restrideArrayImpl<uint8_t> (1, shape, xpSrcStrides, src, dst);
  case 2: return restrideArrayImpl<uint16_t>(2, shape, xpSrcStrides, src, dst);
  case 4: return restrideArrayImpl<uint32_t>(4, shape, xpSrcStrides, src, dst);
  case 8: return restrideArrayImpl<uint64_t>(8, shape, xpSrcStrides, src, dst);
  default: llvm_unreachable("unsupported elementBytewidth");
  }
  // clang-format on
}

} // namespace onnx_mlir
