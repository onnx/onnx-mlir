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

#include <algorithm>

// SSE2 is part of the x86-64 ABI baseline (guaranteed present on every
// x86-64 target, unlike AVX2/AVX-512), so this needs no runtime CPU-feature
// dispatch. Every other target (e.g. AArch64, z/OS/s390x) keeps the portable
// scalar path below unchanged.
#if defined(__x86_64__) || defined(_M_X64)
#define ONNX_MLIR_HAS_SSE2_BYTE_TRANSPOSE 1
#include <emmintrin.h>
#endif

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

  const bool containsZeroDim =
      llvm::any_of(reshapedShape, [](int64_t val) { return val == 0; });
  if (areStridesContiguous(shape, strides) || containsZeroDim)
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

#if ONNX_MLIR_HAS_SSE2_BYTE_TRANSPOSE
// Transposes a 16x16 byte block: `src` points at column 0, with consecutive
// columns `rows` bytes apart; `dst` points at row 0, with consecutive rows
// `columns` bytes apart. Uses the standard SSE2 unpack "butterfly" network:
// stages of unpacklo/unpackhi at doubling granularity (1, 2, 4, 8 bytes)
// transpose the data, but leave the 16 result registers in riffle-shuffle
// order rather than natural row order (a well-known property of this class
// of network) -- `perm` below undoes that on the store.
inline void transpose16x16Bytes(
    const char *src, int64_t rows, char *dst, int64_t columns) {
  __m128i in[16];
  for (int c = 0; c < 16; ++c)
    in[c] = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(src + (int64_t)c * rows));

  __m128i a[16];
  for (int i = 0; i < 8; ++i) {
    a[2 * i] = _mm_unpacklo_epi8(in[2 * i], in[2 * i + 1]);
    a[2 * i + 1] = _mm_unpackhi_epi8(in[2 * i], in[2 * i + 1]);
  }
  __m128i b[16];
  for (int i = 0; i < 4; ++i) {
    b[4 * i + 0] = _mm_unpacklo_epi16(a[4 * i + 0], a[4 * i + 2]);
    b[4 * i + 1] = _mm_unpackhi_epi16(a[4 * i + 0], a[4 * i + 2]);
    b[4 * i + 2] = _mm_unpacklo_epi16(a[4 * i + 1], a[4 * i + 3]);
    b[4 * i + 3] = _mm_unpackhi_epi16(a[4 * i + 1], a[4 * i + 3]);
  }
  __m128i c[16];
  for (int i = 0; i < 2; ++i) {
    c[8 * i + 0] = _mm_unpacklo_epi32(b[8 * i + 0], b[8 * i + 4]);
    c[8 * i + 1] = _mm_unpackhi_epi32(b[8 * i + 0], b[8 * i + 4]);
    c[8 * i + 2] = _mm_unpacklo_epi32(b[8 * i + 1], b[8 * i + 5]);
    c[8 * i + 3] = _mm_unpackhi_epi32(b[8 * i + 1], b[8 * i + 5]);
    c[8 * i + 4] = _mm_unpacklo_epi32(b[8 * i + 2], b[8 * i + 6]);
    c[8 * i + 5] = _mm_unpackhi_epi32(b[8 * i + 2], b[8 * i + 6]);
    c[8 * i + 6] = _mm_unpacklo_epi32(b[8 * i + 3], b[8 * i + 7]);
    c[8 * i + 7] = _mm_unpackhi_epi32(b[8 * i + 3], b[8 * i + 7]);
  }
  __m128i d[16];
  for (int i = 0; i < 8; ++i) {
    d[i] = _mm_unpacklo_epi64(c[i], c[i + 8]);
    d[i + 8] = _mm_unpackhi_epi64(c[i], c[i + 8]);
  }
  static constexpr int perm[16] = {
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
  for (int slot = 0; slot < 16; ++slot)
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(dst + (int64_t)perm[slot] * columns),
        d[slot]);
}
#endif // ONNX_MLIR_HAS_SSE2_BYTE_TRANSPOSE

bool restrideByteMatrixTranspose(ArrayRef<int64_t> shape,
    ArrayRef<int64_t> srcStrides, ArrayRef<char> src,
    MutableArrayRef<char> dst) {
  int64_t rows = 0;
  int64_t columns = 0;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] == 1)
      continue;
    if (shape[axis] <= 0)
      return false;
    if (rows == 0 && srcStrides[axis] == 1)
      rows = shape[axis];
    else if (rows != 0 && columns == 0 && srcStrides[axis] == rows)
      columns = shape[axis];
    else
      return false;
  }
  if (columns == 0 || src.size() != static_cast<uint64_t>(rows) * columns ||
      dst.size() != src.size())
    return false;

#if ONNX_MLIR_HAS_SSE2_BYTE_TRANSPOSE
  // Tile size is the SSE2 register width (16 bytes): a hardware constant
  // that applies uniformly to every shape, not a per-model tuned parameter.
  constexpr int64_t tileSize = 16;
  const int64_t rowTiledEnd = rows - (rows % tileSize);
  const int64_t columnTiledEnd = columns - (columns % tileSize);
  for (int64_t rowBase = 0; rowBase < rowTiledEnd; rowBase += tileSize)
    for (int64_t columnBase = 0; columnBase < columnTiledEnd;
        columnBase += tileSize)
      transpose16x16Bytes(src.data() + columnBase * rows + rowBase, rows,
          dst.data() + rowBase * columns + columnBase, columns);
  // Ragged remainder: rows/columns not divisible by the tile size.
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t column = (row < rowTiledEnd ? columnTiledEnd : 0);
        column < columns; ++column)
      dst[row * columns + column] = src[column * rows + row];
  for (int64_t row = rowTiledEnd; row < rows; ++row)
    for (int64_t column = 0; column < columnTiledEnd; ++column)
      dst[row * columns + column] = src[column * rows + row];
#else
  constexpr int64_t tileSize = 32;
  for (int64_t rowBase = 0; rowBase < rows; rowBase += tileSize) {
    const int64_t rowEnd = std::min(rowBase + tileSize, rows);
    for (int64_t columnBase = 0; columnBase < columns; columnBase += tileSize) {
      const int64_t columnEnd = std::min(columnBase + tileSize, columns);
      for (int64_t row = rowBase; row < rowEnd; ++row)
        for (int64_t column = columnBase; column < columnEnd; ++column)
          dst[row * columns + column] = src[column * rows + row];
    }
  }
#endif // ONNX_MLIR_HAS_SSE2_BYTE_TRANSPOSE
  return true;
}

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
  if (elementBytewidth == 1 &&
      restrideByteMatrixTranspose(shape, xpSrcStrides, src, dst))
    return;
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
