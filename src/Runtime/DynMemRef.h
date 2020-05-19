#ifdef __cplusplus
#pragma once

#include <cstdint>
#include <numeric>
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
  T elem(std::vector<int64_t> idxs) const {
    // Ignore the extent of the leading dimension, strides calculation never
    // uses the extent of the leading dimension.
    std::vector<int64_t> sizesVec(sizes + 1, sizes + rank);
    sizesVec.push_back(1);

    std::vector<int64_t> dimStrides;
    std::partial_sum(sizesVec.rbegin(), sizesVec.rend(), dimStrides.rbegin(),
        std::multiplies<int64_t>());
    int64_t elemOffset = std::inner_product(
        idxs.begin(), idxs.end(), dimStrides.begin(), (int64_t)0);

    T *typedPtr = (T *)data;
    return typedPtr[elemOffset];
  }

  template <typename T>
  T &elem(std::vector<int64_t> idxs) {
    // Ignore the extent of the leading dimension, strides calculation never
    // uses the extent of the leading dimension.
    std::vector<int64_t> sizesVec(sizes + 1, sizes + rank);
    sizesVec.push_back(1);

    std::vector<int64_t> dimStrides(sizesVec.size());
    std::partial_sum(sizesVec.rbegin(), sizesVec.rend(), dimStrides.rbegin(),
        std::multiplies<int64_t>());
    int64_t elemOffset = std::inner_product(
        idxs.begin(), idxs.end(), dimStrides.begin(), (int64_t)0);

    T *typedPtr = (T *)data;
    return typedPtr[elemOffset];
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
#endif
