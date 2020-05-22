#ifdef __cplusplus
#pragma once

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
struct DynMemRef {
  void *data;
  void *alignedData;
  INDEX_TYPE offset;

  unsigned int rank;
  INDEX_TYPE *sizes;
  int64_t *strides;

#ifdef __cplusplus
  DynMemRef(int _rank);

  template <typename T>
  static DynMemRef *create(std::vector<int64_t> _sizes) {
    auto dmr = new DynMemRef(_sizes.size());
    dmr->rank = _sizes.size();
    dmr->sizes = (INDEX_TYPE *)malloc(dmr->rank * sizeof(INDEX_TYPE));
    std::copy(_sizes.begin(), _sizes.end(), dmr->sizes);

    dmr->strides = (int64_t *)malloc(dmr->rank * sizeof(int64_t));
    auto computedStrides = dmr->computeStrides();
    std::copy(computedStrides.begin(), computedStrides.end(), dmr->strides);

    dmr->data = malloc(dmr->size() * sizeof(T));
    dmr->alignedData = dmr->data;

    return dmr;
  }

  int64_t size() const;

  std::vector<int64_t> computeStrides() const {
    // Ignore the extent of the leading dimension, strides calculation
    // never uses the extent of the leading dimension.
    std::vector<int64_t> sizesVec(sizes + 1, sizes + rank);
    sizesVec.push_back(1);

    std::vector<int64_t> dimStrides(rank);
    std::partial_sum(sizesVec.rbegin(), sizesVec.rend(), dimStrides.rbegin(),
        std::multiplies<>());
    return dimStrides;
  }

  int64_t computeOffset(std::vector<int64_t> &idxs) const {
    auto dimStrides = computeStrides();
    int64_t elemOffset = std::inner_product(
        idxs.begin(), idxs.end(), dimStrides.begin(), (int64_t)0);
    return elemOffset;
  }

  template <typename T>
  T &elem(std::vector<int64_t> idxs) {
    int64_t elemOffset = computeOffset(idxs);
    T *typedPtr = (T *)data;
    return typedPtr[elemOffset];
  }

  template <typename T>
  T &elem(int64_t idx) {
    T *typedPtr = (T *)data;
    return typedPtr[idx];
  }

  template <typename T>
  T *typedPtr() {
    return (T *)data;
  }

  std::vector<std::vector<int64_t>> cart_product(
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
      s = move(r);
    }
    return s;
  }

  std::vector<std::vector<int64_t>> indexSet() {
    std::vector<std::vector<int64_t>> dimWiseIdxSet;
    for (auto dimSize : std::vector<int64_t>(sizes, sizes + rank)) {
      std::vector<int64_t> dimIdxSet(dimSize);
      std::iota(std::begin(dimIdxSet), std::end(dimIdxSet), 0);
      dimWiseIdxSet.emplace_back(dimIdxSet);
    }
    return cart_product(dimWiseIdxSet);
  }
#endif
};

#ifdef __cplusplus
// Ordered DynMemRef Dictionary is a data structure for wrapping the input
// dynmemrefs so that they can be addressed both by index and by name.
struct OrderedDynMemRefDict;

#else
typedef struct DynMemRef DynMemRef;
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
DynMemRef *createDynMemRef(int rank);

// Get the i-th dynmemref from orderedDict.
DynMemRef *getDynMemRef(OrderedDynMemRefDict *orderedDict, int i);

// Set the i-th dynmemref in orderedDict to be dynMemRef.
void setDynMemRef(
    OrderedDynMemRefDict *tensorDict, int idx, DynMemRef *dynMemRef);

// Get data pointer from dynMemRef.
void *getData(DynMemRef *dynMemRef);

// Set data pointer for dynMemRef.
void setData(DynMemRef *dynMemRef, void *data);

// Get algined data pointer from dynMemRef.
void *getAlignedData(DynMemRef *);

// Set aligned data pointer for dynMemRef.
void setAlignedData(DynMemRef *, void *);

// Get ptr to sizes array.
INDEX_TYPE *getSizes(DynMemRef *);

// Get ptr to strides array.
int64_t *getStrides(DynMemRef *);

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
DynMemRef *getRndRealDmr(std::vector<int64_t> sizes, T lb = -1.0, T ub = 1.0) {
  // Will be used to obtain a seed for the random number engine
  std::random_device rd;
  // Standard mersenne_twister_engine seeded with rd()
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(lb, ub);
  auto dmr = DynMemRef::create<T>(sizes);
  auto ptr = (float *)dmr->data;
  std::generate(ptr, ptr + dmr->size(), [&]() { return dis(gen); });
  return dmr;
}

template <typename T>
inline bool assertDmrClose(
    DynMemRef *a, DynMemRef *b, float rtol = 1e-5, float atol = 1e-5) {

  // Compare shape.
  auto aShape = std::vector<int64_t>(a->sizes, a->sizes + a->rank);
  auto bShape = std::vector<int64_t>(b->sizes, b->sizes + b->rank);
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
