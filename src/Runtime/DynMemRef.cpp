#include <cassert>
#include <map>
#include <string>
#include <vector>

#include "DynMemRef.h"

DynMemRef::DynMemRef(int _rank) {
  rank = _rank;
  sizes = (INDEX_TYPE *)malloc(rank * sizeof(INDEX_TYPE));
  strides = (int64_t *)malloc(rank * sizeof(int64_t));
}

// An ordered dynamic MemRef dictionary.
// The goal is to support accessing dynamic memory ref by name and by index.
// Currently, only accessing by index is supported.
struct OrderedDynMemRefDict {
  std::map<std::string, DynMemRef *> tensorDict;
  std::vector<std::string> orderedNames;
};

int numDynMemRefs(OrderedDynMemRefDict *dict) {
  return dict->orderedNames.size();
}

OrderedDynMemRefDict *createOrderedDynMemRefDict() {
  return new OrderedDynMemRefDict();
}

DynMemRef *createDynMemRef(int rank) { return new DynMemRef(rank); }

DynMemRef *getDynMemRef(OrderedDynMemRefDict *tensorDict, int idx) {
  return tensorDict->tensorDict[tensorDict->orderedNames[idx]];
}

void setDynMemRef(
    OrderedDynMemRefDict *tensorDict, int idx, DynMemRef *tensor) {
  if (tensorDict->orderedNames.size() <= idx)
    tensorDict->orderedNames.resize(idx + 1);

  // The dynamic memref is essentially anonymous, since we are storing it by
  // indexed position.
  // TODO: can use random string as names to reduce chance of collision.
  auto unique_name = std::to_string(idx);
  assert(tensorDict->tensorDict.count(unique_name) == 0 &&
         "duplicate dynamic mem ref name");

  tensorDict->orderedNames[idx] = unique_name;
  tensorDict->tensorDict[tensorDict->orderedNames[idx]] = tensor;
}

void *getData(DynMemRef *dynMemRef) { return dynMemRef->data; }

void setData(DynMemRef *dynMemRef, void *dataPtr) { dynMemRef->data = dataPtr; }

void *getAlignedData(DynMemRef *dynMemRef) { return dynMemRef->alignedData; }

void setAlignedData(DynMemRef *dynMemRef, void *dataPtr) {
  dynMemRef->alignedData = dataPtr;
}

INDEX_TYPE *getSizes(DynMemRef *dynMemRef) { return dynMemRef->sizes; }

void setSizes(DynMemRef *dynMemRef, INDEX_TYPE *sizes) {
  for (int i = 0; i < dynMemRef->rank; i++)
    dynMemRef->sizes[i] = sizes[i];
}

int64_t *getStrides(DynMemRef *dynMemRef) { return dynMemRef->strides; }

void setStrides(DynMemRef *dynMemRef, int64_t *strides) {
  for (int i = 0; i < dynMemRef->rank; i++)
    dynMemRef->sizes[i] = strides[i];
}
