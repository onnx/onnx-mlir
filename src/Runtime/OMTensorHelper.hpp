/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OMTensorHelper.hpp - OMTensor Helper Func header ----------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMTensor C++ helper functions. At some
// point, this file needs to be merged into the OMTensor.h along with other C++
// APIs operating on OMTensor.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_TENSOR_HELPER_H
#define ONNX_MLIR_TENSOR_HELPER_H

#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

struct OMTensor;

/* Helper function to compute cartesian product */
static inline std::vector<std::vector<int64_t>> CartProduct(
    const std::vector<std::vector<int64_t>> &v) {
  std::vector<std::vector<int64_t>> s = {{}};
  for (const auto &u : v) {
    std::vector<std::vector<int64_t>> r;
    for (const auto &x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = std::move(r);
  }
  return s;
}

/* Helper function to compute data strides from sizes */
static inline std::vector<int64_t> computeStridesFromShape(
    const int64_t *dataSizes, int rank) {
  // Shift dimension sizes one to the left, fill in the vacated rightmost
  // element with 1; this gets us a vector that'll be more useful for computing
  // strides of memory access along each dimension using prefix product (aka
  // partial_sum with a multiply operator below). The intuition is that the size
  // of the leading dimension does not matter when computing strides.
  std::vector<int64_t> sizesVec(dataSizes + 1, dataSizes + rank);
  sizesVec.push_back(1);

  std::vector<int64_t> dimStrides(rank);
  std::partial_sum(sizesVec.rbegin(), sizesVec.rend(), dimStrides.rbegin(),
      std::multiplies<>());
  return dimStrides;
}

/* Helper function to compute linear offset from a multi-dimensional index array
 */
static inline int64_t computeElemOffset(
    const int64_t *dataStrides, int rank, const std::vector<int64_t> &indexes) {
  std::vector<int64_t> dimStrides(dataStrides, dataStrides + rank);
  return inner_product(indexes.begin(), indexes.end(), dimStrides.begin(), 0);
}

/* Helper function to print a vector with delimiter */
template <typename T>
static inline void printVector(const std::vector<T> &vec,
    const std::string &_delimiter = ",", std::ostream &stream = std::cout) {
  std::string delimiter;
  for (const auto &elem : vec) {
    stream << delimiter << elem;
    delimiter = _delimiter;
  }
}

/**
 * OMTensor creator with data sizes and element type
 *
 * @param dataSizes, data sizes array
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 * Create a full OMTensor of data type T and shape dataSizes, with all
 * data fields initialized to proper values and data pointers malloc'ed.
 */
template <typename T>
OMTensor *omTensorCreateWithShape(const std::vector<int64_t> &dataSizes);

/**
 * omDefineSeed.
 * When called, the random number generator for omTensorCreateWithRandomData
 * will be seeded exactly once. The seed is randomly generated when ignoreSeed
 * is nonnull; otherwise the input seedValue is used.
 *
 * @param seed input seed.
 * @param hasSeedValue when nonzero, this function uses the provided seed.
 * Otherwise, this function defines its own random seed.
 * @return return the seed that was used.
 *
 */
unsigned int omDefineSeed(unsigned int seed, unsigned int hasSeedValue);

/**
 * OMTensor creator with data sizes, element type and random data
 *
 * @param dataSizes, data sizes array
 * @param lbound (optional), lower bound of the random distribution
 * @param ubound (optional), upper bound of the random distribution
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 * Create a full OMTensor like what omTensorCreateWithShape does
 * and also fill the OMTensor data buffer with randomly generated
 * real numbers from a uniform distribution between lbound and ubound.
 */
template <typename T>
OMTensor *omTensorCreateWithRandomData(
    const std::vector<int64_t> &dataSizes, T lbound = -1.0, T ubound = 1.0);

/**
 * OMTensor data element getter by offset
 *
 * @param omt, pointer to the OMTensor
 * @param indexes, multi-dimensional index array of the element
 * @return typed element by reference at the offset computed by the index array.
 */
template <typename T>
T &omTensorGetElem(const OMTensor *omt, const std::vector<int64_t> &indexes);

/**
 * OMTensor data element getter by index
 *
 * @param omt, pointer to the OMTensor
 * @param index, index of the element
 * @return typed element by reference at the linear offset.
 */
template <typename T>
T &omTensorGetElemByOffset(const OMTensor *omt, int64_t index);

/**
 * OMTensor strides computation
 *
 * @param omt, pointer to the OMTensor
 * @return data strides of the OMTensor computed from the data sizes.
 */
std::vector<int64_t> omTensorComputeStridesFromShape(const OMTensor *omt);

/**
 * OMTensor linear offset computation
 *
 * @param omt, pointer to the OMTensor
 * @param indexes, multi-dimensional index array
 * @return linear offset.
 */
int64_t omTensorComputeElemOffset(
    const OMTensor *omt, const std::vector<int64_t> &indexes);

/**
 * OMTensor index set computation
 *
 * @param omt, pointer to the OMTensor
 * @return index set (i.e., all valid multi-dimensional array indexes
 *         that can be used to access this OMTensor's constituent elements)
 *         for the whole OMTensor.
 */
std::vector<std::vector<int64_t>> omTensorComputeIndexSet(const OMTensor *omt);

/**
 * OMTensor "distance" computation
 *
 * @param a, 1st OMTensor
 * @param b, 2nd OMTensor
 * @param rtol (optional), relative difference tolerance
 * @param atol (optional), absolute difference tolerance
 * @return true if both relative and absolute difference are within the
 *         specified tolerance, respectively, false otherwise.
 */
template <typename T>
bool omTensorAreTwoOmtsClose(
    const OMTensor *a, const OMTensor *b, float rtol = 1e-5, float atol = 1e-5);
#endif