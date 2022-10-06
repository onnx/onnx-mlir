/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------------- PerfectHash.cpp - Perfect Hash Table ----------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the implementation of a perfect hash table.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/PerfectHash.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <numeric>
#include <string>
#include <vector>

#define DEBUG_TYPE "perfect_hash"

namespace onnx_mlir {

class Utilities {
public:
  // Perform a 32-bit FNV (Fowler-Noll-Vo) hash on the given string.
  static inline uint32_t hash(uint32_t hval, llvm::StringRef str) {
    constexpr uint32_t prime = 0x01000193;
    hval = (hval == 0) ? prime : hval;

    for (const char c : str) {
      hval *= prime;
      hval ^= c;
    }
    return hval;
  }

  // Adaptation of 32-bit FNV for int64_t values.
  static inline uint32_t hash(uint32_t hval, int64_t val) {
    return hash(hval, std::to_string(val));
  }

  // Extracts the keys of the given map.
  template <typename KeyType, typename ValueType>
  static llvm::SmallVector<KeyType> extractKeys(
      const std::map<KeyType, ValueType> &map) {
    llvm::SmallVector<KeyType> keys;
    for (const auto &entry : map)
      keys.push_back(entry.first);
    return keys;
  }

  // Generate the integers in the range [0 .. max-1].
  static llvm::SmallVector<uint32_t> range(uint32_t max) {
    llvm::SmallVector<uint32_t> range(max);
    std::iota(range.begin(), range.end(), 0);
    return range;
  }

  // Generate the integers in the range [min .. max-1].
  static llvm::SmallVector<uint32_t> range(uint32_t min, uint32_t max) {
    llvm::SmallVector<uint32_t> range(max - min);
    std::iota(range.begin(), range.end(), min);
    return range;
  }

  // Generate the integers [min, min+step, ...].
  static llvm::SmallVector<uint32_t> range(
      int32_t min, int32_t max, int32_t step) {
    llvm::SmallVector<uint32_t> range;
    int32_t nElems = (max - min) / step;
    if (nElems < 1)
      return range;

    range.resize(nElems);
    int32_t num = min;
    std::generate_n(range.begin(), nElems, [&num, step]() {
      int32_t res = num;
      num += step;
      return res;
    });
    return range;
  }

  template <typename T>
  static void print(const llvm::SmallVectorImpl<T> &V,
      const llvm::StringRef name, llvm::raw_ostream &os) {
    os << name << ": [ ";
    for (const T &elem : V)
      os << elem << ", ";
    os << "]\n";
  }

  template <typename KeyType, typename ValueType>
  static void print(const std::map<KeyType, ValueType> &M,
      const llvm::StringRef name, llvm::raw_ostream &os) {
    os << name << " : {";
    for (const auto &entry : M)
      os << "'" << entry.first << "': " << entry.second << ", ";
    os << "}\n";
  }
};

template <typename KeyTy, typename ValueTy>
PerfectHash<KeyTy, ValueTy>::PerfectHash(const std::map<KeyTy, ValueTy> &dict)
    : dict(dict) {
  assert(!dict.empty() && "Dictionary should not be empty");
  size_t size = dict.size();
  G.resize(size, 0);
  V.resize(size, -1);
  createPerfectHash();
}

// Note: KeyTy is expected to be either a char* or a int64_t.
// TODO: add a trait to ensure template cannot be instantiated with an
// unexpected type.
template <typename KeyTy, typename ValueTy>
void PerfectHash<KeyTy, ValueTy>::createPerfectHash() {
  LLVM_DEBUG({ Utilities::print(dict, "dict", llvm::dbgs()); });

  // Step 1: place all of the keys into buckets.
  size_t size = dict.size();
  llvm::SmallVector<KeyTy> keys = Utilities::extractKeys<KeyTy, ValueTy>(dict);
  LLVM_DEBUG({ Utilities::print(keys, "keys", llvm::dbgs()); });

  llvm::SmallVector<llvm::SmallVector<KeyTy>> buckets(size);
  for (const KeyTy &key : keys)
    buckets[Utilities::hash(0, key) % size].push_back(key);

  // Step 2: Sort the buckets and process the ones with the most items first.
  llvm::sort(buckets, [](const llvm::SmallVectorImpl<KeyTy> &v1,
                          const llvm::SmallVectorImpl<KeyTy> &v2) {
    return v1.size() > v2.size();
  });

  uint32_t biMax = 0;
  for (uint32_t bi : Utilities::range(size)) {
    LLVM_DEBUG(llvm::dbgs() << "bi=" << bi << "\n");
    biMax = bi;
    llvm::SmallVector<KeyTy> &bucket = buckets[bi];
    if (bucket.size() <= 1)
      break;

    int32_t hval = 1;
    size_t item = 0;
    llvm::SmallVector<uint32_t> slots;

    // Repeatedly try different hash values until we find a hash function that
    // places all items in the bucket into free slots.
    while (item < bucket.size()) {
      uint32_t slot = Utilities::hash(hval, bucket[item]) % size;
      if (V[slot] != -1 ||
          std::find(slots.begin(), slots.end(), slot) != slots.end()) {
        hval++;
        item = 0;
        slots.clear();
      } else {
        slots.push_back(slot);
        item++;
      }
    }

    G[Utilities::hash(0, bucket[0]) % size] = hval;
    for (uint32_t i : Utilities::range(bucket.size()))
      V[slots[i]] = dict.at(bucket[i]);

    LLVM_DEBUG({ Utilities::print(G, "G", llvm::dbgs()); });
    LLVM_DEBUG({ Utilities::print(V, "V", llvm::dbgs()); });
  }

  // Place remaining buckets (containing a single entry) into a free slot. Use
  // a negative value of hval to indicate this.
  llvm::SmallVector<uint32_t> freeList;
  for (uint32_t i : Utilities::range(size))
    if (V[i] == -1)
      freeList.push_back(i);

  LLVM_DEBUG(Utilities::print(freeList, "freeList", llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "biMax: " << biMax << "\n");

  for (uint32_t i : Utilities::range(biMax, size)) {
    llvm::SmallVector<KeyTy> &bucket = buckets[i];
    if (bucket.size() == 0)
      break;

    uint32_t slot = freeList.back();
    freeList.pop_back();

    // Subtract one to ensure it's negative even if the zeroeth slot was used.
    G[Utilities::hash(0, bucket[0]) % size] = -(int32_t)slot - 1;
    V[slot] = dict.at(bucket[0]);

    LLVM_DEBUG({ Utilities::print(G, "G", llvm::dbgs()); });
    LLVM_DEBUG({ Utilities::print(V, "V", llvm::dbgs()); });
  }
}

template class PerfectHash<int64_t, int32_t>;
template class PerfectHash<llvm::StringRef, int32_t>;

} // namespace onnx_mlir
