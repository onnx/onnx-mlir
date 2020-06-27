//===------------ RtMemRef.h - Dynamic MemRef Implementation -------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of Dynamic MemRef data structures and helper
// functions.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#else
#include <stdint.h>
#endif

typedef int64_t INDEX_TYPE;

// This is a dynamic version of memref.
// The same struct can be used to represent memrefs of
// all ranks and type combinations.
// We will refer to it as a DMR (Dynamic MemRef).
struct RtMemRef {

  // Pointer to the raw memory space allocated to host the DMR content. This
  // pointer should only be acessed for memory management purposes, not for
  // reading DMR content.
  void *data;

  // Pointer to the properly aligned array of elements stored in this Dmr.
  void *alignedData;

  // Distance between the start of the raw memory space and the first element of
  // the DMR content.
  INDEX_TYPE offset;

  // Number of dimensions of the array represented by the DMR.
  unsigned int rank;

  // An array recording the per-dimension sizes of the array represented by the
  // DMR.
  INDEX_TYPE *sizes;

  // An array recording the per-dimension strides of the array represented by
  // the DMR.
  int64_t *strides;

  // Refer to TensorProto_DataType at
  // https://github.com/onnx/onnx/blob/cc2230603422bae893d5bc900d2d773ab34400a4/onnx/onnx-ml.proto#L451
  // for enum value interpretation.
  unsigned int dtype;
#ifdef __cplusplus
  explicit RtMemRef(int _rank);

  // Create a full DMR of type T and shape _sizes, with all data fields
  // initialized to proper values and data pointers malloc'ed.
  template <typename T>
  static RtMemRef *create(std::vector<INDEX_TYPE> _sizes) {
    auto dmr = new RtMemRef(_sizes.size());
    dmr->offset = 0;
    dmr->rank = _sizes.size();
    dmr->sizes = (INDEX_TYPE *)malloc(dmr->rank * sizeof(INDEX_TYPE));
    std::copy(_sizes.begin(), _sizes.end(), dmr->sizes);

    dmr->strides = (int64_t *)malloc(dmr->rank * sizeof(int64_t));
    auto computedStrides = dmr->computeStridesFromSizes();
    std::copy(computedStrides.begin(), computedStrides.end(), dmr->strides);

    dmr->data = malloc(dmr->size() * sizeof(T));
    dmr->alignedData = dmr->data;

    return dmr;
  }

  // Access an element (by reference) at index position idxs.
  template <typename T>
  T &elem(std::vector<INDEX_TYPE> idxs) {
    INDEX_TYPE elemOffset = computeOffset(idxs);
    T *typedPtr = (T *)data;
    return typedPtr[elemOffset];
  }

  // Access an element (by reference) at *flattened* index position idx.
  template <typename T>
  T &elem(INDEX_TYPE idx) {
    T *typedPtr = (T *)data;
    return typedPtr[idx];
  }

  // Get a typed ptr to the data content of the DMR.
  template <typename T>
  T *typedPtr() {
    return (T *)data;
  }

  // Get how many elements are stored in DMR, as implied by its shape.
  INDEX_TYPE size() const;

  // Helper function to compute strides of access along each dimensions from its
  // shape.
  std::vector<int64_t> computeStridesFromSizes() const;

  // Compute flattened array idx from a multi-dimensional array idx.
  INDEX_TYPE computeOffset(std::vector<INDEX_TYPE> &idxs) const;

  // Get the index set (i.e., all valid multi-dimensional array indexes that can
  // be used to access this DMR's constituent elements).
  std::vector<std::vector<INDEX_TYPE>> indexSet() const;

  ~RtMemRef();
#endif
};

#ifdef __cplusplus
// Ordered RtMemRef Dictionary is a data structure for wrapping the input
// dynmemrefs so that they can be addressed both by index and by name.
struct OrderedDynMemRefDict;

#else
typedef struct RtMemRef RtMemRef;
typedef struct _OrderedDynMemRefDict OrderedDynMemRefDict;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Get number of dynamic memrefs in OrderedDynMemRefDict dict.
int numDynMemRefs(OrderedDynMemRefDict *dict);

// Create an ordered dynamic memref dictionary.
OrderedDynMemRefDict *createOrderedDynMemRefDict();

// Get how many dynamic memrefs are in dict.
int64_t getSize(OrderedDynMemRefDict *dict);

// Create a dynmemref with a certain rank.
RtMemRef *createDynMemRef(int rank);

// Get the i-th dynmemref from orderedDict.
RtMemRef *getDynMemRef(OrderedDynMemRefDict *orderedDict, int i);

// Set the i-th dynmemref in orderedDict to be dynMemRef.
void setDynMemRef(
        OrderedDynMemRefDict *tensorDict, int idx, RtMemRef *dynMemRef);

// Get data pointer from dynMemRef.
void *getData(RtMemRef *dynMemRef);

// Set data pointer for dynMemRef.
void setData(RtMemRef *dynMemRef, void *data);

// Get algined data pointer from dynMemRef.
void *getAlignedData(RtMemRef *);

// Set aligned data pointer for dynMemRef.
void setAlignedData(RtMemRef *, void *);

// Get the data type enum value of the dynMemRef.
int getDType(RtMemRef *dynMemRef);

// Set the data type enum value of the dynMemRef.
void setDType(RtMemRef *dynMemRef, int onnxType);

// Get the rank of the dynMemRef.
unsigned int getRank(RtMemRef *dynMemRef);

// Get ptr to sizes array.
INDEX_TYPE *getSizes(RtMemRef *);

// Set the sizes array (by copying size values from array `sizes`).
void setSizes(RtMemRef *, INDEX_TYPE *sizes);

// Get ptr to strides array.
int64_t *getStrides(RtMemRef *);

// Set the strides array (by copying stride values from array `strides`).
void setStrides(RtMemRef *, int64_t *strides);

#ifdef __cplusplus
}

template <typename T>
void printVector(std::vector<T> vec, std::string _delimiter = ",",
    std::ostream &stream = std::cout) {
  std::string delimiter;
  for (const auto &elem : vec) {
    stream << delimiter << elem;
    delimiter = _delimiter;
  }
}

template <typename T>
RtMemRef *getRndRealDmr(
    std::vector<INDEX_TYPE> sizes, T lb = -1.0, T ub = 1.0) {
  // Will be used to obtain a seed for the random number engine
  std::random_device rd;
  // Standard mersenne_twister_engine seeded with rd()
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(lb, ub);
  auto dmr = RtMemRef::create<T>(sizes);
  auto ptr = (T *)dmr->data;
  std::generate(ptr, ptr + dmr->size(), [&]() { return dis(gen); });
  return dmr;
}

template <typename T>
inline bool isDmrClose(
        RtMemRef *a, RtMemRef *b, float rtol = 1e-5, float atol = 1e-5) {

  // Compare shape.
  auto aShape = std::vector<INDEX_TYPE>(a->sizes, a->sizes + a->rank);
  auto bShape = std::vector<INDEX_TYPE>(b->sizes, b->sizes + b->rank);
  if (aShape != bShape) {
    std::cerr << "Shape mismatch ";
    printVector(aShape, ",", std::cerr);
    std::cerr << " != ";
    printVector(bShape, ",", std::cerr);
    return false;
  }

  // Compute absolute difference, verify it's within tolerable range.
  std::vector<T> absoluteDiff(a->size());
  std::transform(a->typedPtr<T>(), a->typedPtr<T>() + a->size(),
      b->typedPtr<T>(), absoluteDiff.begin(), std::minus<>());
  std::transform(absoluteDiff.begin(), absoluteDiff.end(), absoluteDiff.begin(),
      static_cast<T (*)(T)>(&std::abs));
  bool atolSatisfied = std::all_of(
      absoluteDiff.begin(), absoluteDiff.end(), [&](T a) { return a < atol; });

  // Compute relative difference, verify it's within tolerable range.
  std::vector<T> relativeDiff(a->size());
  std::transform(absoluteDiff.begin(), absoluteDiff.end(), a->typedPtr<T>(),
      relativeDiff.begin(), std::divides<>());
  bool rtolSatisfied = std::all_of(
      relativeDiff.begin(), relativeDiff.end(), [&](T a) { return a < rtol; });

  if (atolSatisfied && rtolSatisfied) {
    return true;
  } else {
    // Figure out where and what went wrong, this can be slow; but hopefully we
    // don't need this often.
    for (const auto &idx : a->indexSet()) {
      T aElem = a->elem<T>(idx);
      T bElem = b->elem<T>(idx);
      auto elmAbsDiff = std::abs(aElem - bElem);
      auto withinRtol = (elmAbsDiff / aElem < rtol);
      auto withinAtol = (elmAbsDiff < atol);
      if (!withinRtol || !withinAtol) {
        std::cerr << "a[";
        printVector(idx, ",", std::cerr);
        std::cerr << "] = " << aElem << " != ";
        std::cerr << "b[";
        printVector(idx, ",", std::cerr);
        std::cerr << "] = " << bElem << std::endl;
      }
    }
    return false;
  }
}
#endif

// Will transition from DynMemRef to RtMemRef soon.
typedef RtMemRef DynMemRef;