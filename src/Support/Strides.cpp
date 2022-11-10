/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- Strides.cpp -----------------------------===//
//
// Strides helper functions.
//
//===----------------------------------------------------------------------===//

#include "src/Support/Strides.hpp"

#include "src/Support/Arrays.hpp"

#include "mlir/IR/BuiltinTypeInterfaces.h"

using namespace mlir;

namespace onnx_mlir {

int64_t getStridesNumElements(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
  if (ShapedType::getNumElements(shape) == 0)
    return 0;
  assert(shape.size() >= strides.size());
  int64_t last = 0;
  for (int a = shape.size() - 1, s = strides.size() - 1; s >= 0; --a, --s)
    last += (shape[a] - 1) * strides[s];
  return last + 1;
}

size_t getStridesPosition(
    ArrayRef<int64_t> indices, ArrayRef<int64_t> strides) {
  assert(indices.size() >= strides.size());
  size_t pos = 0;
  for (int a = indices.size() - 1, s = strides.size() - 1; s >= 0; --a, --s)
    pos += indices[a] * strides[s];
  return pos;
}

bool areStridesContiguous(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
  unsigned rank = shape.size();
  assert(rank >= strides.size());
  int skip = rank - strides.size();
  auto leadingOnes =
      shape.take_while([](int64_t dimSize) { return dimSize == 1; });
  if (unsigned(skip) != leadingOnes.size())
    return false;
  int64_t mult = 1;
  for (int axis = rank - 1; axis >= skip; --axis) {
    int64_t dimSize = shape[axis];
    if (strides[axis - skip] != (dimSize == 1 ? 0 : mult))
      return false;
    mult *= dimSize;
  }
  return true;
}

SmallVector<int64_t, 4> getDefaultStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> strides;
  int64_t rank = shape.size();
  if (rank == 0)
    return strides;
  int64_t skip = 0;
  while (shape[skip] == 1) {
    ++skip;
    if (skip == rank)
      return strides;
  }
  strides.resize_for_overwrite(rank - skip);
  int64_t mult = 1;
  for (int64_t axis = rank - 1; axis >= skip; --axis) {
    int64_t dimSize = shape[axis];
    strides[axis - skip] = dimSize == 1 ? 0 : mult;
    mult *= dimSize;
  }
  return strides;
}

Optional<SmallVector<int64_t, 4>> transposeStrides(ArrayRef<int64_t> shape,
    ArrayRef<int64_t> strides, ArrayRef<uint64_t> perm) {
  // TODO: refine logic to figure out strides in more situations
  if (strides != makeArrayRef(getDefaultStrides(shape)))
    return None;
  return transposeDims(strides, perm);
}

Optional<SmallVector<int64_t, 4>> reshapeStrides(ArrayRef<int64_t> shape,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> reshapedShape) {
  // TODO: refine logic to figure out strides in more situations
  if (strides != makeArrayRef(getDefaultStrides(shape)))
    return None;
  return getDefaultStrides(reshapedShape);
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

SmallVector<int64_t, 4> unflattenIndex(
    ArrayRef<int64_t> shape, int64_t flatIndex) {
  SmallVector<int64_t, 4> indices;
  int64_t rank = shape.size();
  if (rank > 0) {
    indices.resize_for_overwrite(rank);
    for (int64_t axis = rank - 1; axis >= 1; --axis) {
      int64_t dimSize = shape[axis];
      assert(dimSize > 0 && "cannot unflatten shape with zeros");
      int64_t rem = flatIndex % dimSize;
      flatIndex /= dimSize;
      indices[axis] = rem;
    }
    assert(flatIndex < shape[0]);
    indices[0] = flatIndex;
  }
  return indices;
}

namespace {

// Uses the same algorithm as transformAndRestrideTwoWideArrays but is
// sufficiently different to be reimplemented here without code reuse.
template <typename T>
void restrideArrayImpl(ArrayRef<int64_t> shape, Strided<ArrayRef<T>> src,
    Strided<MutableArrayRef<T>> dst) {
  size_t rank = shape.size();
  assert(src.strides.size() == rank && "src strides must match rank");
  assert(dst.strides.size() == rank && "dst strides must match rank");
  auto traverse = [=](size_t axis, size_t srcPos, size_t dstPos,
                      const auto &recurse) -> void {
    if (axis == rank) {
      dst.data[dstPos] = src.data[srcPos];
    } else {
      size_t srcStride = src.strides[axis];
      size_t dstStride = dst.strides[axis];
      size_t dimSize = shape[axis];
      for (size_t i = 0; i < dimSize; ++i) {
        recurse(axis + 1, srcPos, dstPos, recurse);
        srcPos += srcStride;
        dstPos += dstStride;
      }
    }
  };
  traverse(0, 0, 0, traverse);
}

// clang-format off
template <unsigned Bytewidth>
using BitcastType =
    std::conditional_t<Bytewidth == 1, std::uint8_t,
    std::conditional_t<Bytewidth == 2, std::uint16_t,
    std::conditional_t<Bytewidth == 4, std::uint32_t,
    std::conditional_t<Bytewidth == 8, std::uint64_t,
    void>>>>;
// clang-format on

template <unsigned Bytewidth>
struct BytewidthToken {
  constexpr BytewidthToken() {}
  constexpr operator unsigned() const { return Bytewidth; }
};

template <typename Action, typename... Args>
auto dispatchByBytewidth(unsigned bytewidth, Action &&act, Args &&...args) {
  // clang-format off
  switch (bytewidth) {
  case 1: return act(BytewidthToken<1>{}, std::forward<Args>(args)...);
  case 2: return act(BytewidthToken<2>{}, std::forward<Args>(args)...);
  case 4: return act(BytewidthToken<4>{}, std::forward<Args>(args)...);
  case 8: return act(BytewidthToken<8>{}, std::forward<Args>(args)...);
  default: llvm_unreachable("unsupported bytewidth");
  }
  // clang-format on
}

} // namespace

void restrideArray(unsigned bytewidth, ArrayRef<int64_t> shape,
    Strided<ArrayRef<char>> src, Strided<MutableArrayRef<char>> dst) {
  auto expandedSrcStrides = expandStrides(src.strides, shape);
  dispatchByBytewidth(bytewidth, [&](auto staticBytewidth) {
    using T = BitcastType<staticBytewidth>;
    Strided<ArrayRef<T>> srcT{expandedSrcStrides, castArrayRef<T>(src.data)};
    Strided<MutableArrayRef<T>> dstT{
        dst.strides, castMutableArrayRef<T>(dst.data)};
    restrideArrayImpl<T>(shape, srcT, dstT);
  });
}

} // namespace onnx_mlir