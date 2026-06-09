/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OMTensorHelper.cpp - OMTensor C++ debug helpers -----------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// Random-data tensor creation helpers compiled as a separate library
// (OMDebugRuntime) so they can be linked alongside a statically compiled model
// without conflicting with the model's bundled cruntime C symbols.
//
// These functions use only the public C API (omTensorCreateWithOwnership etc.)
// and never access OMTensor struct internals directly.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "OnnxMlirRuntime.h"
#include "src/Runtime/OMTensorHelper.hpp"
#include "src/Support/SmallFPConversion.h"

// =============================================================================
// RNG state — owned here, shared via omDefineSeed.

static unsigned int omUseOneSeed = 0;
static std::mt19937 omRandomGenerator(0);

unsigned int omDefineSeed(unsigned int seed, unsigned int hasSeedValue) {
  if (!hasSeedValue) {
    std::random_device rd;
    seed = rd();
  }
  omUseOneSeed = 1;
  omRandomGenerator.seed(seed);
  return seed;
}

// =============================================================================
// Internal: fill a typed buffer with random values in [lbound, ubound].
// Uses the shared omRandomGenerator.  Re-seeds if omUseOneSeed is not set.

template <typename T>
static void fillBufferWithRandom(T *buf, int64_t n, T lbound, T ubound) {
  if (!omUseOneSeed) {
    std::random_device rd;
    omRandomGenerator.seed(rd());
  }
  if (lbound == ubound) {
    std::fill(buf, buf + n, lbound);
    return;
  }
  if constexpr (std::is_integral<T>::value) {
    if constexpr (sizeof(T) <= 4) {
      std::uniform_int_distribution<int32_t> dis(
          static_cast<int32_t>(lbound), static_cast<int32_t>(ubound));
      std::generate(buf, buf + n,
          [&]() { return static_cast<T>(dis(omRandomGenerator)); });
    } else {
      std::uniform_int_distribution<int64_t> dis(
          static_cast<int64_t>(lbound), static_cast<int64_t>(ubound));
      std::generate(buf, buf + n,
          [&]() { return static_cast<T>(dis(omRandomGenerator)); });
    }
  } else {
    std::uniform_real_distribution<T> dis(lbound, ubound);
    std::generate(buf, buf + n, [&]() { return dis(omRandomGenerator); });
  }
}

// =============================================================================
// Template omTensorCreateWithRandomData<T> — private to this file.
// Called only by the non-template dispatch overload below.
// Uses pure C API — no struct member access.

template <typename T>
static OMTensor *omTensorCreateWithRandomData(
    const std::vector<int64_t> &shape, T lbound, T ubound) {
  int64_t numElems = 1;
  for (auto d : shape)
    numElems *= d;
  T *buf = (T *)malloc(numElems * sizeof(T));
  if (!buf)
    return nullptr;
  fillBufferWithRandom<T>(buf, numElems, lbound, ubound);
  OM_DATA_TYPE dtype =
      OM_DATA_TYPE_CPP_TO_ONNX.at(std::string(typeid(T).name()));
  int64_t rank = (int64_t)shape.size();
  OMTensor *t = omTensorCreateWithOwnership(
      buf, const_cast<int64_t *>(shape.data()), rank, dtype, /*owning=*/1);
  if (!t)
    free(buf);
  return t;
}

// Explicit instantiations.
template OMTensor *omTensorCreateWithRandomData<bool>(
    const std::vector<int64_t> &, bool, bool);
template OMTensor *omTensorCreateWithRandomData<int8_t>(
    const std::vector<int64_t> &, int8_t, int8_t);
template OMTensor *omTensorCreateWithRandomData<uint8_t>(
    const std::vector<int64_t> &, uint8_t, uint8_t);
template OMTensor *omTensorCreateWithRandomData<int16_t>(
    const std::vector<int64_t> &, int16_t, int16_t);
template OMTensor *omTensorCreateWithRandomData<uint16_t>(
    const std::vector<int64_t> &, uint16_t, uint16_t);
template OMTensor *omTensorCreateWithRandomData<int32_t>(
    const std::vector<int64_t> &, int32_t, int32_t);
template OMTensor *omTensorCreateWithRandomData<uint32_t>(
    const std::vector<int64_t> &, uint32_t, uint32_t);
template OMTensor *omTensorCreateWithRandomData<int64_t>(
    const std::vector<int64_t> &, int64_t, int64_t);
template OMTensor *omTensorCreateWithRandomData<uint64_t>(
    const std::vector<int64_t> &, uint64_t, uint64_t);
template OMTensor *omTensorCreateWithRandomData<float>(
    const std::vector<int64_t> &, float, float);
template OMTensor *omTensorCreateWithRandomData<double>(
    const std::vector<int64_t> &, double, double);

// Forward declaration — defined below with the other float16/string helpers.
static OMTensor *omTensorCreateFloat16WithRandomData(
    const std::vector<int64_t> &shape, float lbound, float ubound);

// =============================================================================
// Non-template runtime-dispatch overload.

OMTensor *omTensorCreateWithRandomData(const std::vector<int64_t> &shape,
    OM_DATA_TYPE omType, double lbound, double ubound) {
  switch (omType) {
  case ONNX_TYPE_BOOL:
    // Clamp bounds to {false, true}: values > 0.5 → true, ≤ 0.5 → false.
    return omTensorCreateWithRandomData<bool>(
        shape, lbound > 0.5, ubound > 0.5);
  case ONNX_TYPE_INT8:
    return omTensorCreateWithRandomData<int8_t>(
        shape, (int8_t)lbound, (int8_t)ubound);
  case ONNX_TYPE_UINT8:
    return omTensorCreateWithRandomData<uint8_t>(
        shape, (uint8_t)lbound, (uint8_t)ubound);
  case ONNX_TYPE_INT16:
    return omTensorCreateWithRandomData<int16_t>(
        shape, (int16_t)lbound, (int16_t)ubound);
  case ONNX_TYPE_UINT16:
    return omTensorCreateWithRandomData<uint16_t>(
        shape, (uint16_t)lbound, (uint16_t)ubound);
  case ONNX_TYPE_INT32:
    return omTensorCreateWithRandomData<int32_t>(
        shape, (int32_t)lbound, (int32_t)ubound);
  case ONNX_TYPE_UINT32:
    return omTensorCreateWithRandomData<uint32_t>(
        shape, (uint32_t)lbound, (uint32_t)ubound);
  case ONNX_TYPE_INT64:
    return omTensorCreateWithRandomData<int64_t>(
        shape, (int64_t)lbound, (int64_t)ubound);
  case ONNX_TYPE_UINT64:
    return omTensorCreateWithRandomData<uint64_t>(
        shape, (uint64_t)lbound, (uint64_t)ubound);
  case ONNX_TYPE_FLOAT:
    return omTensorCreateWithRandomData<float>(
        shape, (float)lbound, (float)ubound);
  case ONNX_TYPE_DOUBLE:
    return omTensorCreateWithRandomData<double>(shape, lbound, ubound);
  case ONNX_TYPE_FLOAT16:
    return omTensorCreateFloat16WithRandomData(
        shape, (float)lbound, (float)ubound);
  case ONNX_TYPE_STRING: {
    // Generate random integer strings whose values fall in [lbound, ubound].
    // The integers are converted to their decimal string representation.
    if (!omUseOneSeed) {
      std::random_device rd;
      omRandomGenerator.seed(rd());
    }
    int64_t numElems = 1;
    for (auto d : shape)
      numElems *= d;
    // Create strings into a vector of strings.
    std::uniform_int_distribution<int> dist((int)lbound, (int)ubound);
    std::vector<std::string> strings((size_t)numElems);
    for (int64_t e = 0; e < numElems; ++e)
      strings[e] = std::to_string(dist(omRandomGenerator));
    // Converts the strings in a proper omTensor data of strings, where all
    // pointers and actual strings are co-located in a single buffer.
    void *buf = omTensorBuildStringBuffer(strings);
    if (!buf)
      return nullptr;
    int64_t rank = (int64_t)shape.size();
    OMTensor *t = omTensorCreateWithOwnership(
        buf, const_cast<int64_t *>(shape.data()), rank, ONNX_TYPE_STRING, 1);
    if (!t)
      free(buf);
    return t;
  }
  default:
    fprintf(stderr,
        "omTensorCreateWithRandomData: OM_DATA_TYPE %d is not currently "
        "supported for random data generation.\n",
        (int)omType);
    return nullptr;
  }
}

// =============================================================================
// Float16 and string helpers.

static OMTensor *omTensorCreateFloat16WithRandomData(
    const std::vector<int64_t> &shape, float lbound, float ubound) {
  if (!omUseOneSeed) {
    std::random_device rd;
    omRandomGenerator.seed(rd());
  }
  int64_t numElems = 1;
  for (auto d : shape)
    numElems *= d;
  uint16_t *buf = (uint16_t *)malloc(numElems * sizeof(uint16_t));
  if (!buf)
    return nullptr;
  std::uniform_real_distribution<float> dist(lbound, ubound);
  for (int64_t e = 0; e < numElems; ++e)
    buf[e] = om_f32_to_f16(dist(omRandomGenerator));
  int64_t rank = (int64_t)shape.size();
  OMTensor *t = omTensorCreateWithOwnership(
      buf, const_cast<int64_t *>(shape.data()), rank, ONNX_TYPE_FLOAT16, 1);
  if (!t)
    free(buf);
  return t;
}

// Create Sequences of zozCount Ones, followed by Zeros. Count of -1 means each
// sequence length has a random number of ones.
OMTensor *omTensorCreateSozData(
    const std::vector<int64_t> &shape, OM_DATA_TYPE omType, int64_t sozCount) {
  if (shape.empty())
    return nullptr;
  int64_t innerDim = shape.back();
  int64_t numRows = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i)
    numRows *= shape[i];
  int64_t numElems = numRows * innerDim;
  int64_t elemSize = (int64_t)OM_DATA_TYPE_SIZE[omType];
  if (elemSize <= 0)
    return nullptr;

  void *buf = calloc((size_t)numElems, (size_t)elemSize);
  if (!buf)
    return nullptr;

  // Only seed and construct the distribution when sozCount == -1 (random mode).
  std::uniform_int_distribution<int64_t> dist(
      1, (innerDim > 1) ? innerDim - 1 : 1);
  if (sozCount < 0 && !omUseOneSeed) {
    std::random_device rd;
    omRandomGenerator.seed(rd());
  }
  int64_t fixedN = (sozCount >= 0) ? std::min(sozCount, innerDim) : 0;

  // Generic lambda: fill the first count elements of a typed pointer with 1.
  // n is passed as a parameter rather than captured so the lambda can be
  // defined before the loop where n is computed.
  auto fillOnes = [](auto *p, int64_t count) {
    using T = std::remove_reference_t<decltype(*p)>;
    std::fill(p, p + count, T(1));
  };

  for (int64_t row = 0; row < numRows; ++row) {
    int64_t n = (sozCount < 0) ? dist(omRandomGenerator) : fixedN;
    char *rowPtr = (char *)buf + row * innerDim * elemSize;
    switch (omType) {
    case ONNX_TYPE_BOOL:
      fillOnes((bool *)rowPtr, n);
      break;
    case ONNX_TYPE_INT8:
      fillOnes((int8_t *)rowPtr, n);
      break;
    case ONNX_TYPE_UINT8:
      fillOnes((uint8_t *)rowPtr, n);
      break;
    case ONNX_TYPE_INT16:
      fillOnes((int16_t *)rowPtr, n);
      break;
    case ONNX_TYPE_UINT16:
      fillOnes((uint16_t *)rowPtr, n);
      break;
    case ONNX_TYPE_INT32:
      fillOnes((int32_t *)rowPtr, n);
      break;
    case ONNX_TYPE_UINT32:
      fillOnes((uint32_t *)rowPtr, n);
      break;
    case ONNX_TYPE_INT64:
      fillOnes((int64_t *)rowPtr, n);
      break;
    case ONNX_TYPE_UINT64:
      fillOnes((uint64_t *)rowPtr, n);
      break;
    case ONNX_TYPE_FLOAT:
      fillOnes((float *)rowPtr, n);
      break;
    case ONNX_TYPE_DOUBLE:
      fillOnes((double *)rowPtr, n);
      break;
    default:
      free(buf);
      return nullptr;
    }
  }

  int64_t rank = (int64_t)shape.size();
  OMTensor *t = omTensorCreateWithOwnership(
      buf, const_cast<int64_t *>(shape.data()), rank, omType, /*owning=*/1);
  if (!t)
    free(buf);
  return t;
}

void *omTensorBuildStringBuffer(const std::vector<std::string> &strings) {
  size_t n = strings.size();
  size_t totalStrLen = 0;
  for (const auto &s : strings)
    totalStrLen += s.length() + 1;
  void *buf = malloc(sizeof(char *) * n + totalStrLen);
  if (!buf)
    return nullptr;
  char **ptrArray = (char **)buf;
  char *strData = (char *)buf + sizeof(char *) * n;
  for (size_t i = 0; i < n; ++i) {
    size_t len = strings[i].length();
    memcpy(strData, strings[i].data(), len);
    strData[len] = '\0';
    ptrArray[i] = strData;
    strData += len + 1;
  }
  return buf;
}
