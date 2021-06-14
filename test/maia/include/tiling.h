
#pragma once

#include <assert.h>
#include <gsl/span>
#include <limits>
#include <vector>

template <typename flt_t>
using FloatMatrix = std::vector<std::vector<flt_t>>;

// Convert matrix data to tile array layout by reading continuous tiles
template <typename flt_t, size_t nativeMatrixSize>
std::vector<flt_t> ConvertMatrixToTileArrays(FloatMatrix<flt_t> &mat) {
  size_t nRows = mat.size();
  size_t nCols = mat.at(0).size();
  std::vector<flt_t> result(
      nRows * nCols, std::numeric_limits<flt_t>::quiet_NaN());
  size_t offset = 0;
  for (size_t rTile = 0; rTile < nRows / nativeMatrixSize; ++rTile) {
    for (size_t cTile = 0; cTile < nCols / nativeMatrixSize; ++cTile) {
      for (size_t i = 0; i < nativeMatrixSize; ++i) {
        std::copy(mat.at(rTile * nativeMatrixSize + i).begin() +
                      cTile * nativeMatrixSize,
            mat.at(rTile * nativeMatrixSize + i).begin() +
                (cTile + 1) * nativeMatrixSize,
            result.begin() + offset);
        offset += nativeMatrixSize;
      }
    }
  }

  return result;
}

template <typename flt_t, size_t nativeMatrixSize>
std::vector<flt_t> ConvertMatrixToTileArrays(
    std::vector<FloatMatrix<flt_t>> &mat) {
  std::vector<flt_t> result;
  for (int i = 0; i < mat.size(); ++i) {
    auto segment = ConvertMatrixToTileArrays<flt_t, nativeMatrixSize>(mat[i]);
    result.insert(result.end(), segment.begin(), segment.end());
  }
  return result;
}

template <typename flt_t, size_t nativeMatrixSize>
std::vector<flt_t> ConvertMatrixToTileArrays(
    std::vector<std::vector<FloatMatrix<flt_t>>> &mat) {
  std::vector<flt_t> result;
  for (int i = 0; i < mat.size(); ++i) {
    auto segment = ConvertMatrixToTileArrays<flt_t, nativeMatrixSize>(mat[i]);
    result.insert(result.end(), segment.begin(), segment.end());
  }
  return result;
}

inline size_t compute_size(std::vector<size_t> &shape) {
  size_t size = 1;
  for (auto dim : shape)
    size *= dim;
  return size;
}

template <typename flt_t, size_t nativeMatrixSize>
std::vector<flt_t> RowMajorToTileMajor(
    gsl::span<flt_t> raw_data_untiled, std::vector<size_t> &shape) {
  assert(compute_size(shape) == raw_data_untiled.size());
  std::vector<flt_t> result(raw_data_untiled.size());

  if (shape.size() > 2) {
    size_t offset = 0;
    for (int i = 0; i < shape[0]; ++i) {
      auto inner_shape = std::vector<size_t>{shape.begin() + 1, shape.end()};
      auto size = compute_size(inner_shape);
      auto segment = RowMajorToTileMajor<flt_t, nativeMatrixSize>(
          gsl::span<flt_t>(&raw_data_untiled[0] + offset, size), inner_shape);
      std::copy(segment.begin(), segment.end(), result.begin() + offset);
      offset += size;
    }
    return result;
  }

  assert(shape.size() == 2);

  size_t nRows = shape[0];
  size_t nCols = shape[1];

  assert(nRows % nativeMatrixSize == 0);
  assert(nCols % nativeMatrixSize == 0);

  size_t offset = 0;
  for (size_t rTile = 0; rTile < nRows / nativeMatrixSize; ++rTile) {
    for (size_t cTile = 0; cTile < nCols / nativeMatrixSize; ++cTile) {
      for (size_t i = 0; i < nativeMatrixSize; ++i) {
        auto copy_begin =
            &raw_data_untiled[(rTile * nativeMatrixSize + i) * nCols +
                              cTile * nativeMatrixSize];
        std::copy(
            copy_begin, copy_begin + nativeMatrixSize, result.begin() + offset);
        offset += nativeMatrixSize;
      }
    }
  }

  return result;
}

template <typename flt_t, size_t nativeMatrixSize>
std::vector<flt_t> TileMajorToRowMajor(
    gsl::span<flt_t> raw_data_tiled, std::vector<size_t> &shape) {
  assert(compute_size(shape) == raw_data_tiled.size());
  std::vector<flt_t> result(raw_data_tiled.size());

  if (shape.size() > 2) {
    size_t offset = 0;
    for (int i = 0; i < shape[0]; ++i) {
      auto inner_shape = std::vector<size_t>{shape.begin() + 1, shape.end()};
      auto size = compute_size(inner_shape);
      auto segment = TileMajorToRowMajor<flt_t, nativeMatrixSize>(
          gsl::span<flt_t>(&raw_data_tiled[0] + offset, size), inner_shape);
      std::copy(segment.begin(), segment.end(), result.begin() + offset);
      offset += size;
    }
    return result;
  }

  assert(shape.size() == 2);

  size_t nRows = shape[0];
  size_t nCols = shape[1];

  assert(nRows % nativeMatrixSize == 0);
  assert(nCols % nativeMatrixSize == 0);

  const auto hTiles = nRows / nativeMatrixSize;
  const auto wTiles = nCols / nativeMatrixSize;

  size_t tileOffset = 0;
  for (size_t h = 0; h < hTiles; ++h) {
    for (size_t w = 0; w < wTiles; ++w) {
      size_t tileIndex = 0;
      for (size_t i = 0; i < nativeMatrixSize; ++i) {
        for (size_t j = 0; j < nativeMatrixSize; ++j) {
          result[(h * nativeMatrixSize + i) * nCols + w * nativeMatrixSize +
                 j] = raw_data_tiled[tileOffset + tileIndex];
          tileIndex++;
        }
      }

      tileOffset += nativeMatrixSize * nativeMatrixSize;
    }
  }

  return result;
}

// Converts from tile array layout to matrix layout
template <typename flt_t, size_t nativeMatrixSize>
const FloatMatrix<flt_t> ConvertTileArraysToMatrix(
    gsl::span<const flt_t> tiles, const size_t hMat, const size_t wMat) {
  FloatMatrix<flt_t> mat(hMat, std::vector<flt_t>(wMat, .0f));
  const auto hTiles = hMat / nativeMatrixSize;
  const auto wTiles = wMat / nativeMatrixSize;

  size_t tileOffset = 0;
  for (size_t h = 0; h < hTiles; ++h) {
    for (size_t w = 0; w < wTiles; ++w) {
      size_t tileIndex = 0;
      for (size_t i = 0; i < nativeMatrixSize; ++i) {
        for (size_t j = 0; j < nativeMatrixSize; ++j) {
          mat.at(h * nativeMatrixSize + i).at(w * nativeMatrixSize + j) =
              tiles.at(tileOffset + tileIndex);
          tileIndex++;
        }
      }

      tileOffset += nativeMatrixSize * nativeMatrixSize;
    }
  }

  return mat;
}

template <typename flt_t, size_t nativeMatrixSize>
std::vector<FloatMatrix<flt_t>> ConvertTileArraysToMatrix(
    gsl::span<flt_t> tiles, const size_t z, const size_t hMat,
    const size_t wMat) {
  std::vector<FloatMatrix<flt_t>> result(z);
  auto start = &tiles[0];
  for (int i = 0; i < z; ++i) {
    auto segment = gsl::span<flt_t>(start, hMat * wMat);
    result[i] =
        ConvertTileArraysToMatrix<flt_t, nativeMatrixSize>(segment, hMat, wMat);
    start += segment.size();
  }

  return result;
}

template <typename flt_t, size_t nativeMatrixSize>
std::vector<std::vector<FloatMatrix<flt_t>>> ConvertTileArraysToMatrix(
    gsl::span<flt_t> tiles, const size_t z2, const size_t z, const size_t hMat,
    const size_t wMat) {
  std::vector<std::vector<FloatMatrix<flt_t>>> result(z2);
  auto start = &tiles[0];
  for (int i = 0; i < z2; ++i) {
    auto segment = gsl::span<flt_t>(start, z * hMat * wMat);
    result[i] = ConvertTileArraysToMatrix<flt_t, nativeMatrixSize>(
        segment, z, hMat, wMat);
    start += segment.size();
  }

  return result;
}
