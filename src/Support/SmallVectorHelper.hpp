/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- SmallVectorHelper.hpp - Helper functions llvm::SmallVector -----===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for taking subsets of llvm::SmallVector.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SMALL_VECTOR_HELPER_H
#define ONNX_MLIR_SMALL_VECTOR_HELPER_H

#include "llvm/ADT/SmallVector.h"

//===----------------------------------------------------------------------===//
// Select the first few elements of a vector, until "untilNum" (inclusively)
// Negative numbers count from the back of the vector.

// Note: because it is inclusively, it is impossible to have an empty list.

template <typename T, unsigned N>
llvm::SmallVector<T, N> firstFew(mlir::ValueRange vec, int64_t untilNum) {
  llvm::SmallVector<T, N> res;
  int64_t size = vec.size();
  if (untilNum < 0)
    untilNum += size;
  // If untilNum<0...  we get an empty vector, that is ok.
  assert(untilNum < size && "out of bound");
  for (int64_t i = 0; i <= untilNum; ++i)
    res.emplace_back(vec[i]);
  return res;
}

template <typename T, unsigned N>
llvm::SmallVector<T, N> firstFew(mlir::ArrayRef<T> vec, int64_t untilNum) {
  llvm::SmallVector<T, N> res;
  int64_t size = vec.size();
  if (untilNum < 0)
    untilNum += size;
  // If untilNum<0...  we get an empty vector, that is ok.
  assert(untilNum < size && "out of bound");
  for (int64_t i = 0; i <= untilNum; ++i)
    res.emplace_back(vec[i]);
  return res;
}

template <typename T, unsigned N>
llvm::SmallVector<T, N> firstFew(
    llvm::SmallVectorImpl<T> &vec, int64_t untilNum) {
  llvm::SmallVector<T, N> res;
  int64_t size = vec.size();
  if (untilNum < 0)
    untilNum += size;
  // If untilNum<0...  we get an empty vector, that is ok.
  assert(untilNum < size && "out of bound");
  for (int64_t i = 0; i <= untilNum; ++i)
    res.emplace_back(vec[i]);
  return res;
}

//===----------------------------------------------------------------------===//
// Select the last few elements of a vector, from "untilNum" (inclusively)
// Negative numbers count from the back of the vector.

template <typename T, unsigned N>
llvm::SmallVector<T, N> lastFew(mlir::ValueRange vec, int64_t fromNum) {
  llvm::SmallVector<T, N> res;
  int64_t size = vec.size();
  if (fromNum < 0)
    fromNum += size;
  // If fromNum>= size...  we get an empty vector, that is ok.
  assert(fromNum >= 0 && "out of bound");
  for (int64_t i = fromNum; i < size; ++i)
    res.emplace_back(vec[i]);
  return res;
}

template <typename T, unsigned N>
llvm::SmallVector<T, N> lastFew(mlir::ArrayRef<T> vec, int64_t fromNum) {
  llvm::SmallVector<T, N> res;
  int64_t size = vec.size();
  if (fromNum < 0)
    fromNum += size;
  // If fromNum>= size...  we get an empty vector, that is ok.
  assert(fromNum >= 0 && "out of bound");
  for (int64_t i = fromNum; i < size; ++i)
    res.emplace_back(vec[i]);
  return res;
}

template <typename T, unsigned N>
llvm::SmallVector<T, N> lastFew(
    llvm::SmallVectorImpl<T> &vec, int64_t fromNum) {
  llvm::SmallVector<T, N> res;
  int64_t size = vec.size();
  if (fromNum < 0)
    fromNum += size;
  // If fromNum>= size...  we get an empty vector, that is ok.
  assert(fromNum >= 0 && "out of bound");
  for (int64_t i = fromNum; i < size; ++i)
    res.emplace_back(vec[i]);
  return res;
}

#endif
