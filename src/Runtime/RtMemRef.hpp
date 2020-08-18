//===-------- RtMemRef.hpp - RtMemRef Implementation header --------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of RtMemRef and data structures and
// helper functions.
//
//===----------------------------------------------------------------------===//
#pragma once
#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

/* Typically, MemRefs in MLIR context are used as a compile-time constructs.
 * Information such as element type and rank of the data payload is statically
 * encoded, meaning that they are determined and fixed at compile-time. This
 * presents significant burden for any runtime components trying to interact
 * with the compiled executable.
 *
 * Thus a version of MemRef struct that is amenable to runtime manipulation is
 * provided as a basis for building any runtime-related components providing
 * user-facing programming interfaces. All information are dynamically encoded
 * as members of this struct so that they can be accessed and modified easily
 * during runtime.
 *
 * We will refer to it as a RMF (Runtime MemRef).
 */

struct RtMemRef {
  /**
   * Constructor
   *
   * @param rank, rank of data sizes and strides
   *
   * Create a RtMemRef with specified rank. Memory for data sizes and strides
   * are allocated.
   */
  RtMemRef(int rank) {
    if ((_dataSizes = (INDEX_TYPE *)malloc(rank * sizeof(INDEX_TYPE))) &&
        (_dataStrides = (int64_t *)malloc(rank * sizeof(int64_t)))) {
      _data = NULL;
      _alignedData = NULL;
      _offset = 0;
      _dataType = ONNX_TYPE_UNDEFINED;
      _rank = rank;
      _owningData = false;
    } else {
      throw std::runtime_error(
          "RtMemRef(" + std::to_string(rank) + ") malloc error");
    }
  };

  /**
   * Destructor
   *
   * Destroy the RtMemRef struct.
   */
  ~RtMemRef(void) {
    if (_owningData)
      free(_data);
    free(_dataSizes);
    free(_dataStrides);
  };

  void *_data;            /* data buffer                                   */
  void *_alignedData;     /* aligned data buffer that the rmr indexes.     */
  INDEX_TYPE _offset;     /* offset of 1st element                         */
  INDEX_TYPE *_dataSizes; /* sizes array                                   */
  int64_t *_dataStrides;  /* strides array                                 */
  int _dataType;          /* ONNX data type                                */
  int _rank;              /* rank                                          */
  std::string _name;      /* optional name for named access                */
  bool _owningData;       /* if we malloc _data ourselves, destructor will
                           * free _data, otherwise, if _data is set via setData call,
                           * then free _data is caller's responsibility       */
};

struct RtMemRefList {
  /**
   * Constructor
   *
   * Create an RtMemRefList with specified RtMemRef pointer array
   * and the size of the array
   */
  RtMemRefList(RtMemRef *rmrs[], int n) { _rmrs.assign(&rmrs[0], &rmrs[n]); };

#ifdef RTMEMREF_INTERNAL_API
  /**
   * Constructor
   *
   * Create an empty RtMemRefList for internal API calls.
   */
  RtMemRefList(void){};
#endif

  /**
   * Destructor
   *
   * Destroy the RtMemRefList struct.
   */
  ~RtMemRefList(void) {
    /* Destroy all the RtMemRefs */
    for (int i = 0; i < _rmrs.size(); i++)
      if (_rmrs[i])
        rmrDestroy(_rmrs[i]);
  };

  /* To facilitate user facing API getRmrs, RtMemRefs are kept in a vector
   * that can be quickly returned as an array. A name to index map is used
   * to address ReMemRefs by name.
   */
  std::vector<RtMemRef *> _rmrs; /* RtMemRef vector   */
};

/*------------------------------------------------------- */
/* Internal function used by RtMemRef itself, not exposed */
/*------------------------------------------------------- */

/* Helper function to compute the number of data elements */
static inline INDEX_TYPE getNumOfElems(INDEX_TYPE *dataSizes, int rank) {
  return accumulate(dataSizes, dataSizes + rank, 1, std::multiplies<>());
}

/* Helper function to get the ONNX data type size in bytes */
static inline int getDataTypeSize(int dataType) {
  return dataType < 0 ||
                 dataType >= sizeof(RTMEMREF_DATA_TYPE_SIZE) / sizeof(int)
             ? 0
             : RTMEMREF_DATA_TYPE_SIZE[dataType];
}

#ifdef RTMEMREF_INTERNAL_API

/* Helper function to compute cartisian product */
static inline std::vector<std::vector<INDEX_TYPE>> CartProduct(
    const std::vector<std::vector<INDEX_TYPE>> &v) {
  std::vector<std::vector<INDEX_TYPE>> s = {{}};
  for (const auto &u : v) {
    std::vector<std::vector<INDEX_TYPE>> r;
    for (const auto &x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = move(r);
  }
  return s;
}

/* Helper function to compute data strides from sizes */
static inline std::vector<int64_t> computeStridesFromSizes(
    INDEX_TYPE *dataSizes, int rank) {
  // Shift dimension sizes one to the left, fill in the vacated rightmost
  // element with 1; this gets us a vector that'll be more useful for computing
  // strides of memory access along each dimension using prefix product (aka
  // partial_sum with a multiply operator below). The intuition is that the size
  // of the leading dimension does not matter when computing strides.
  std::vector<int64_t> sizesVec(dataSizes + 1, dataSizes + rank);
  sizesVec.push_back(1);

  std::vector<int64_t> dimStrides(rank);
  partial_sum(sizesVec.rbegin(), sizesVec.rend(), dimStrides.rbegin(),
      std::multiplies<>());
  return dimStrides;
}

/* Helper function to compute linear offset from a multi-dimensional index array
 */
static inline INDEX_TYPE computeElemOffset(
    int64_t *dataStrides, int rank, std::vector<INDEX_TYPE> &indexes) {
  auto dimStrides = std::vector<INDEX_TYPE>(dataStrides, dataStrides + rank);
  INDEX_TYPE elemOffset = inner_product(
      indexes.begin(), indexes.end(), dimStrides.begin(), (INDEX_TYPE)0);
  return elemOffset;
}

/* Helper function to print a vector with delimiter */
template <typename T>
static inline void printVector(std::vector<T> vec, std::string _delimiter = ",",
    std::ostream &stream = std::cout) {
  std::string delimiter;
  for (const auto &elem : vec) {
    stream << delimiter << elem;
    delimiter = _delimiter;
  }
}

#endif
