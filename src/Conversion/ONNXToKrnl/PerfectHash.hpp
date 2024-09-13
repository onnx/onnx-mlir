/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------------- PerfectHash.hpp - Perfect Hash Table ----------------===//
//
// Copyright 2021-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declaration of a perfect hash table.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PERFECT_HASH_H
#define ONNX_MLIR_PERFECT_HASH_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <map>

namespace onnx_mlir {

template <typename T>
struct is_supported {
  enum { value = false };
};

template <>
struct is_supported<int64_t> {
  enum { value = true };
};

template <>
struct is_supported<llvm::StringRef> {
  enum { value = true };
};

template <typename KeyTy, typename ValueTy>
class PerfectHash {
  static_assert(is_supported<KeyTy>::value, "KeyTy not supported");

  // The hash table is defined by G and V.
  llvm::SmallVector<int32_t> G;
  llvm::SmallVector<int32_t> V;
  const std::map<KeyTy, ValueTy> &dict;

public:
  PerfectHash(const std::map<KeyTy, ValueTy> &dict);

  const llvm::SmallVector<int32_t> &getG() const { return G; }
  const llvm::SmallVector<int32_t> &getV() const { return V; }

private:
  // Creates a minimal perfect hash for the dictionary.
  void createPerfectHash();
};

} // namespace onnx_mlir
#endif
