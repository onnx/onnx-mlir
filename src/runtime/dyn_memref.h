#pragma once

#include <cstdint>

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

  DynMemRef(int _rank);
};

extern "C" {

// Ordered DynMemRef Dictionary is a data structure for wrapping the input
// dynmemrefs so that they can be addressed both by index and by name.
struct OrderedDynMemRefDict;

// Get number of dynamic memrefs in OrderedDynMemRefDict dict.
int numDynMemRefs(OrderedDynMemRefDict *dict);

// Create an ordered dynmemref dictionary.
OrderedDynMemRefDict *createOrderedDynMemRefDict();

// Create a dynmemref with a certain rank.
DynMemRef *createDynMemRef(int rank);

// Get the i-th dynmemref from orderedDict.
DynMemRef *getDynMemRef(OrderedDynMemRefDict *orderedDict, int i);

// Set the i-th dynmemref in orderedDict to be dynMemRef.
void setDynMemRef(OrderedDynMemRefDict *tensorDict, int idx,
                  DynMemRef *dynMemRef);

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
}
