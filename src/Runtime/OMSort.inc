/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- OMSort.inc - OMSort C/C++ Implementation --===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ implementation of OMSort.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"
#include "onnx-mlir/Runtime/OnnxDataType.h"
#ifdef __cplusplus
#include "src/Runtime/OMTensorHelper.hpp"
#endif

#include "src/Support/SmallFPConversion.h"

//
// Declare compare functions for data types and sorting directions.
//
// Data-type-specific compare functions are used here for performance reason.
// As background, the performance of this sort function is important for the
// whole model performance (e.g. the dominant part of the Yolov3 model).
// If we use data-type-general compare function, including switch statements
// to check the data type in the compare function, the switch statements are
// executed in the sort and compare loops, we can avoid this overhead by
// using data-type-specific compare functions to check the data type at once
// before entering the loop.
//
// This function is expected to provide "stable" sort that preserve the input
// data order, if the values are the same. Although most sort algorithms (e.g.
// quick sort) are not stable, we use special compare functions to compare two
// values at first, then compare the input orders if the values are the same.
// This comparison functions guarantee the input order among the same values
// and makes sorting algorithms stable.
//
#define compareFunctionBody(typeName, direction, load, dataPtr, idx1p, idx2p)  \
  {                                                                            \
    uint64_t idx1 = *((uint64_t *)(idx1p));                                    \
    uint64_t idx2 = *((uint64_t *)(idx2p));                                    \
    typeName *data = (typeName *)(dataPtr);                                    \
    load(typeName, v1, data[idx1]);                                            \
    load(typeName, v2, data[idx2]);                                            \
    return (direction(v1, v2) || (v1 == v2 && idx1 < idx2)) ? -1 : 1;          \
  }
#if defined(__APPLE__)
typedef int(
    compareFunctionType(void *dataPtr, const void *idx1, const void *idx2));
#define declare_compare_function(fname, typeName, direction, load)             \
  int compare##fname##direction(                                               \
      void *dataPtr, const void *idx1p, const void *idx2p)                     \
      compareFunctionBody(typeName, direction, load, dataPtr, idx1p, idx2p)
#elif defined(_MSC_VER)
typedef int(__cdecl compareFunctionType)(void *, const void *, const void *);
#define declare_compare_function(fname, typeName, direction, load)             \
  int compare##fname##direction(                                               \
      void *dataPtr, const void *idx1p, const void *idx2p)                     \
      compareFunctionBody(typeName, direction, load, dataPtr, idx1p, idx2p)
#else
typedef int(
    compareFunctionType(const void *idx1, const void *idx2, void *dataPtr));
#define declare_compare_function(fname, typeName, direction, load)             \
  int compare##fname##direction(                                               \
      const void *idx1p, const void *idx2p, void *dataPtr)                     \
      compareFunctionBody(typeName, direction, load, dataPtr, idx1p, idx2p)
#endif
#define compareFunction(fname, direction) compare##fname##direction
// clang-format off
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif

#define Load(typeName, to, from) typeName to = from

// Convert f16 elements to f32 for comparison because we don't have logic to
// compare f16 elements directly on all platforms.
// TODO: Use Convert(_Float16, to, from) on supported platforms.
//       Or consider converting the whole tensor to f32, sort as f32, and then
//       convert the sorted tensor back to f16. That may be faster than
//       converting elements for each comparison during sorting.
#define LoadF16AsF32(typeName, to, from) float (to) = om_f16_to_f32(from)

// declare ascending functions
#define Ascending(lhs, rhs) ((lhs) < (rhs))
declare_compare_function(Bool, bool, Ascending, Load)
declare_compare_function(Uint8, uint8_t, Ascending, Load)
declare_compare_function(Int8, int8_t, Ascending, Load)
declare_compare_function(Uint16, uint16_t, Ascending, Load)
declare_compare_function(Int16, int16_t, Ascending, Load)
declare_compare_function(Uint32, uint32_t, Ascending, Load)
declare_compare_function(Int32, int32_t, Ascending, Load)
declare_compare_function(Uint64, uint64_t, Ascending, Load)
declare_compare_function(Int64, int64_t, Ascending, Load)
declare_compare_function(Float, float, Ascending, Load)
declare_compare_function(Double, double, Ascending, Load)
declare_compare_function(Float16, uint16_t, Ascending, LoadF16AsF32)

// declare descending functions
#define Descending(lhs, rhs) ((lhs) > (rhs))
declare_compare_function(Bool, bool, Descending, Load)
declare_compare_function(Uint8, uint8_t, Descending, Load)
declare_compare_function(Int8, int8_t, Descending, Load)
declare_compare_function(Uint16, uint16_t, Descending, Load)
declare_compare_function(Int16, int16_t, Descending, Load)
declare_compare_function(Uint32, uint32_t, Descending, Load)
declare_compare_function(Int32, int32_t, Descending, Load)
declare_compare_function(Uint64, uint64_t, Descending, Load)
declare_compare_function(Int64, int64_t, Descending, Load)
declare_compare_function(Float, float, Descending, Load)
declare_compare_function(Double, double, Descending, Load)
declare_compare_function(Float16, uint16_t, Descending, LoadF16AsF32)

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
// clang-format on

//
// Declare sort functions for sorting algorithms
//
#ifdef __APPLE__
                            typedef void(sortFunctionType(void *base,
                                size_t dataNum, size_t dataSize, void *dataPtr,
                                compareFunctionType compFunc));
#elif defined(_MSC_VER)
                            typedef void(sortFunctionType(void *base,
                                size_t num, size_t width,
                                compareFunctionType compFunc, void *context));
#else
                            typedef void(sortFunctionType(void *base,
                                size_t dataNum, size_t dataSize,
                                compareFunctionType compFunc, void *dataPtr));
#endif

//
// Custom quick sort function for environments not suppirting qsort_r (e.g. zos)
//
#define SWAP_INDEX(a, b)                                                       \
  do {                                                                         \
    uint64_t tmp = (a);                                                        \
    (a) = (b);                                                                 \
    (b) = tmp;                                                                 \
  } while (0)
// stack for index (uint64_t)
typedef struct indexStack {
  uint64_t *stackData;
  int64_t stackSize;
  int64_t stackTop;
} indexStack;

#define STACK_INIT(stack, stackSize)                                           \
  do {                                                                         \
    assert((stackSize) > 0);                                                   \
    (stack).stackData = (uint64_t *)alloca((stackSize) * sizeof(uint64_t));    \
    assert((stack).stackData != NULL);                                         \
    (stack).stackSize = (stackSize);                                           \
    (stack).stackTop = 0;                                                      \
  } while (0)
#define STACK_ISEMPTY(stack) ((stack).stackTop == 0)
#define STACK_PUSH(stack, begin, end)                                          \
  do {                                                                         \
    assert((stack).stackTop <= (stack).stackSize - 2);                         \
    (stack).stackData[((stack).stackTop)++] = (begin);                         \
    (stack).stackData[((stack).stackTop)++] = (end);                           \
  } while (0)
#define STACK_POP(stack, begin, end)                                           \
  do {                                                                         \
    assert((stack).stackTop >= 2);                                             \
    (end) = (stack).stackData[--((stack).stackTop)];                           \
    (begin) = (stack).stackData[--((stack).stackTop)];                         \
  } while (0)
#define STACK_PRINT(stack)                                                     \
  do {                                                                         \
    assert((stack).stackTop >= 0);                                             \
    fprintf(stderr, "Stack: [");                                               \
    for (int64_t i = 0; (i + 1) < (stack).stackTop; i += 2) {                  \
      fprintf(                                                                 \
          stderr, "<%ld:%ld>, ", (stack).stackData[i], (stack).stackData[i + 1]);\
    }                                                                          \
    fprintf(                                                                   \
        stderr, "] (Top=%ld,Size=%ld)\n", (stack).stackTop, (stack).stackSize);\
    fflush(stderr);                                                            \
  } while (0)

static int64_t log2u(uint64_t n) {
  assert(n > 0);
  int64_t b = 0;
  for (; n > 1; b++)
    n = n >> 1;
  return b;
}

// Quick sort patition function
static int64_t quick_sort_partition(void *dataPtr, uint64_t *idx,
    compareFunctionType compFunc, int64_t begin, int64_t end) {
  int64_t i = begin;
  for (int64_t j = begin; j < end; j++) {
#if defined(__APPLE__)
    if (compFunc(dataPtr, idx + j, idx + end) <= 0)
#elif defined(_MSC_VER)
    if (compFunc(dataPtr, idx + j, idx + end) <= 0)
#else
    if (compFunc(idx + j, idx + end, dataPtr) <= 0)
#endif
    {
      SWAP_INDEX(idx[i], idx[j]);
      i++;
    }
  }
  SWAP_INDEX(idx[end], idx[i]);
  return i;
}

// Quick sort main function (custom version)
#ifdef __APPLE__
void quick_sort_custom(void *base, size_t dataNum, size_t dataSize,
    void *dataPtr, compareFunctionType compFunc) {
#else
void quick_sort_custom(void *base, size_t dataNum, size_t dataSize,
    compareFunctionType compFunc, void *dataPtr) {
#endif
  uint64_t *idx = (uint64_t *)base;
  // Calculate the theoritical maximum stack size for index
  int64_t stackSize = (log2u(dataNum + 1) + 2) * 2;
  indexStack stack;
  STACK_INIT(stack, stackSize);
  int64_t begin = 0;
  int64_t end = dataNum - 1;

  // push current job to the stack
  STACK_PUSH(stack, begin, end);

  while (!STACK_ISEMPTY(stack)) {
    // pop current job from the stack
    STACK_POP(stack, begin, end);
    if (begin < end) {
      int64_t pivotIdx =
          quick_sort_partition(dataPtr, idx, compFunc, begin, end);
      // To limit the stack size, push larger partion at first
      if ((pivotIdx - begin) > (end - pivotIdx)) {
        if (begin < pivotIdx - 1)
          STACK_PUSH(stack, begin, pivotIdx - 1);
        if (pivotIdx + 1 < end)
          STACK_PUSH(stack, pivotIdx + 1, end);
      } else {
        if (pivotIdx + 1 < end)
          STACK_PUSH(stack, pivotIdx + 1, end);
        if (begin < pivotIdx - 1)
          STACK_PUSH(stack, begin, pivotIdx - 1);
      }
    }
  }
}

compareFunctionType *getCompareFunction(
    uint64_t ascending, OM_DATA_TYPE dataType) {
  compareFunctionType *compFunc;

  switch (dataType) {
  case ONNX_TYPE_BOOL:
    compFunc = ascending ? compareFunction(Bool, Ascending)
                         : compareFunction(Bool, Descending);
    break;
  case ONNX_TYPE_UINT8:
    compFunc = ascending ? compareFunction(Uint8, Ascending)
                         : compareFunction(Uint8, Descending);
    break;
  case ONNX_TYPE_INT8:
    compFunc = ascending ? compareFunction(Int8, Ascending)
                         : compareFunction(Int8, Descending);
    break;
  case ONNX_TYPE_UINT16:
    compFunc = ascending ? compareFunction(Uint16, Ascending)
                         : compareFunction(Uint16, Descending);
    break;
  case ONNX_TYPE_INT16:
    compFunc = ascending ? compareFunction(Int16, Ascending)
                         : compareFunction(Int16, Descending);
    break;
  case ONNX_TYPE_UINT32:
    compFunc = ascending ? compareFunction(Uint32, Ascending)
                         : compareFunction(Uint32, Descending);
    break;
  case ONNX_TYPE_INT32:
    compFunc = ascending ? compareFunction(Int32, Ascending)
                         : compareFunction(Int32, Descending);
    break;
  case ONNX_TYPE_UINT64:
    compFunc = ascending ? compareFunction(Uint64, Ascending)
                         : compareFunction(Uint64, Descending);
    break;
  case ONNX_TYPE_INT64:
    compFunc = ascending ? compareFunction(Int64, Ascending)
                         : compareFunction(Int64, Descending);
    break;
  case ONNX_TYPE_FLOAT:
    compFunc = ascending ? compareFunction(Float, Ascending)
                         : compareFunction(Float, Descending);
    break;
  case ONNX_TYPE_DOUBLE:
    compFunc = ascending ? compareFunction(Double, Ascending)
                         : compareFunction(Double, Descending);
    break;
  case ONNX_TYPE_FLOAT16:
    compFunc = ascending ? compareFunction(Float16, Ascending)
                         : compareFunction(Float16, Descending);
    break;
  default:
    assert(false && "unexpected data type in getCompareFunction");
  }
  return compFunc;
}

void omTensorSort(OMTensor *orderTensor, const OMTensor *inputTensor,
    uint64_t axis, uint64_t ascending) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  const uint64_t rank = omTensorGetRank(inputTensor);
  assert(rank <= 6 && "omTensorSort assumes rank <= 6");
  assert(axis == (rank - 1) && "omTensorSort assumes axis == (rank - 1)");
  const int64_t *inputShape = omTensorGetShape(inputTensor);
  const int64_t *inputStrides = omTensorGetStrides(inputTensor);
  assert(inputStrides[axis] == 1 && "omTensorSort assumes strides[axis] == 1");
  void *orderPtr = omTensorGetDataPtr(orderTensor);
  uint64_t *order = (uint64_t *)orderPtr;
  void *dataPtr = omTensorGetDataPtr(inputTensor);
  int64_t sort_elems = inputShape[axis];
  // Sorting not necessary for empty array
  if (sort_elems == 0)
    return;

  // Choose the appropriate compare function
  compareFunctionType *compareElements =
      getCompareFunction(ascending, dataType);
  uint64_t datasize = OM_DATA_TYPE_SIZE[dataType];

#if defined(__APPLE__)
  // MacOS supports qsort_r in different API
  sortFunctionType *sortFunc = qsort_r;
#elif defined(_MSC_VER)
#pragma warning(push, 3)
// Newer MSVC warns 4113 instead of 4028 for function signature mismatch.
// Disable both here.
#pragma warning(disable : 4028)
#pragma warning(disable : 4113)
  // Windows supports qsort_s
  sortFunctionType *sortFunc = qsort_s;
#pragma warning(pop)
#elif defined(__linux) || defined(__linux__) || defined(linux)
  // Use standard quick sort in libc
  sortFunctionType *sortFunc = qsort_r;
#else // for environments not supporting quick sort
  sortFunctionType *sortFunc = quick_sort_custom; // custum quick sort
#endif
  // To support input Tensor with various ranks in a uniform way.
  // If the input rank < 6, upgrade the rank to 6 virtually without changing
  // the physical memory layout by inserting length=1 ranks at lower ranks.
  // The 5th axis becomes the sort axis.
  int64_t shape[6] = {1, 1, 1, 1, 1, 1};
  int64_t strides[6] = {0, 0, 0, 0, 0, 0};
  for (uint64_t i = 0; i < rank; i++) {
    shape[i + (6 - rank)] = inputShape[i];
    strides[i + (6 - rank)] = inputStrides[i];
  }

  // Sort the the 6th axis in the outer 5 loops
  for (int dim0 = 0; dim0 < shape[0]; dim0++) {
    for (int dim1 = 0; dim1 < shape[1]; dim1++) {
      for (int dim2 = 0; dim2 < shape[2]; dim2++) {
        for (int dim3 = 0; dim3 < shape[3]; dim3++) {
          for (int dim4 = 0; dim4 < shape[4]; dim4++) {
            uint64_t off = dim0 * strides[0] + dim1 * strides[1] +
                           dim2 * strides[2] + dim3 * strides[3] +
                           dim4 * strides[4];
            void *data = ((char *)dataPtr) + datasize * off;
            uint64_t *idx = order + off;
#if defined(__APPLE__)
            sortFunc((void *)idx, sort_elems, sizeof(uint64_t), data,
                compareElements);
#else
            sortFunc((void *)idx, sort_elems, sizeof(uint64_t), compareElements,
                data);
#endif
          }
        }
      }
    }
  }
  return;
}
