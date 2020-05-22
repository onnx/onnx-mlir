#include <cassert>
#include <map>
#include <string>
#include <vector>

#include "DynMemRef.h"

namespace {
// Helper function to compute cartisian product.
inline static std::vector<std::vector<INDEX_TYPE>> CartProduct(
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
} // namespace

DynMemRef::DynMemRef(int _rank) {
  rank = _rank;
  sizes = (INDEX_TYPE *)malloc(rank * sizeof(INDEX_TYPE));
  strides = (int64_t *)malloc(rank * sizeof(int64_t));
}

INDEX_TYPE DynMemRef::size() const {
  return std::accumulate(sizes, sizes + rank, 1, std::multiplies<>());
}

std::vector<std::vector<INDEX_TYPE>> DynMemRef::indexSet() const {
  std::vector<std::vector<INDEX_TYPE>> dimWiseIdxSet;
  for (auto dimSize : std::vector<INDEX_TYPE>(sizes, sizes + rank)) {
    std::vector<INDEX_TYPE> dimIdxSet(dimSize);
    std::iota(std::begin(dimIdxSet), std::end(dimIdxSet), 0);
    dimWiseIdxSet.emplace_back(dimIdxSet);
  }
  return CartProduct(dimWiseIdxSet);
}

INDEX_TYPE DynMemRef::computeOffset(std::vector<INDEX_TYPE> &idxs) const {
  auto dimStrides = computeStrides();
  INDEX_TYPE elemOffset = std::inner_product(
      idxs.begin(), idxs.end(), dimStrides.begin(), (INDEX_TYPE)0);
  return elemOffset;
}

std::vector<int64_t> DynMemRef::computeStrides() const {
  // Ignore the extent of the leading dimension, strides calculation
  // never uses the extent of the leading dimension.
  std::vector<int64_t> sizesVec(sizes + 1, sizes + rank);
  sizesVec.push_back(1);

  std::vector<int64_t> dimStrides(rank);
  std::partial_sum(sizesVec.rbegin(), sizesVec.rend(), dimStrides.rbegin(),
      std::multiplies<>());
  return dimStrides;
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

int64_t getSize(OrderedDynMemRefDict *dict) {
  return dict->orderedNames.size();
}

void setStrides(DynMemRef *dynMemRef, int64_t *strides) {
  for (int i = 0; i < dynMemRef->rank; i++)
    dynMemRef->sizes[i] = strides[i];
}
