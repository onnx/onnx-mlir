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

#define TRUE 1
#define FALSE 0

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
#if defined(__APPLE__)
typedef int(
    compareFunctionType(void *dataPtr, const void *idx1, const void *idx2));
#define declare_compare_function(fname, typeName, direction, symbol)           \
  int compare##fname##direction(                                               \
      void *dataPtr, const void *idx1p, const void *idx2p) {                   \
    uint64_t idx1 = *((uint64_t *)idx1p);                                      \
    uint64_t idx2 = *((uint64_t *)idx2p);                                      \
    typeName *data = (typeName *)dataPtr;                                      \
    return ((data[idx1] symbol data[idx2]) ||                                  \
               ((data[idx1] == data[idx2]) && (idx1 > idx2)))                  \
               ? 1                                                             \
               : -1;                                                           \
  }
#elif defined(_MSC_VER)
typedef int(__cdecl compareFunctionType)(void *, const void *, const void *);
#define declare_compare_function(fname, typeName, direction, symbol)           \
  int compare##fname##direction(                                               \
      void *dataPtr, const void *idx1p, const void *idx2p) {                   \
    uint64_t idx1 = *((uint64_t *)idx1p);                                      \
    uint64_t idx2 = *((uint64_t *)idx2p);                                      \
    typeName *data = (typeName *)dataPtr;                                      \
    return ((data[idx1] symbol data[idx2]) ||                                  \
               ((data[idx1] == data[idx2]) && (idx1 > idx2)))                  \
               ? 1                                                             \
               : -1;                                                           \
  }
#else
typedef int(
    compareFunctionType(const void *idx1, const void *idx2, void *dataPtr));
#define declare_compare_function(fname, typeName, direction, symbol)           \
  int compare##fname##direction(                                               \
      const void *idx1p, const void *idx2p, void *dataPtr) {                   \
    uint64_t idx1 = *((uint64_t *)idx1p);                                      \
    uint64_t idx2 = *((uint64_t *)idx2p);                                      \
    typeName *data = (typeName *)dataPtr;                                      \
    return ((data[idx1] symbol data[idx2]) ||                                  \
               ((data[idx1] == data[idx2]) && (idx1 > idx2)))                  \
               ? 1                                                             \
               : -1;                                                           \
  }
#endif
#define compareFunction(fname, typeName, direction) compare##fname##direction
// clang-format off
// declare ascending functions
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif
declare_compare_function(Bool, bool, Ascending, >)
declare_compare_function(Uint8, uint8_t, Ascending, >)
declare_compare_function(Int8, int8_t, Ascending, >)
declare_compare_function(Uint16, uint16_t, Ascending, >)
declare_compare_function(Int16, int16_t, Ascending, >)
declare_compare_function(Uint32, uint32_t, Ascending, >)
declare_compare_function(Int32, int32_t, Ascending, >)
declare_compare_function(Uint64, uint64_t, Ascending, >)
declare_compare_function(Int64, int64_t, Ascending, >)
declare_compare_function(Float, float, Ascending, >)
declare_compare_function(Double, double, Ascending, >)
// declare descending functions
declare_compare_function(Bool, bool, Descending, <)
declare_compare_function(Uint8, uint8_t, Descending, <)
declare_compare_function(Int8, int8_t, Descending, <)
declare_compare_function(Uint16, uint16_t, Descending, <)
declare_compare_function(Int16, int16_t, Descending, <)
declare_compare_function(Uint32, uint32_t, Descending, <)
declare_compare_function(Int32, int32_t, Descending, <)
declare_compare_function(Uint64, uint64_t, Descending, <)
declare_compare_function(Int64, int64_t, Descending, <)
declare_compare_function(Float, float, Descending, <)
declare_compare_function(Double, double, Descending, <)
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
#define SWAP_INDEX(a, b)                                                       \
  do {                                                                         \
    uint64_t tmp = (a);                                                        \
    (a) = (b);                                                                 \
    (b) = tmp;                                                                 \
  } while (0)

compareFunctionType *getCompareFunction(
    uint64_t ascending, OM_DATA_TYPE dataType) {
  compareFunctionType *compFunc;

  switch (dataType) {
  case ONNX_TYPE_BOOL:
    compFunc = ascending ? compareFunction(Bool, bool, Ascending)
                         : compareFunction(Bool, bool, Descending);
    break;
  case ONNX_TYPE_UINT8:
    compFunc = ascending ? compareFunction(Uint8, uint8_t, Ascending)
                         : compareFunction(Uint8, uint8_t, Descending);
    break;
  case ONNX_TYPE_INT8:
    compFunc = ascending ? compareFunction(Int8, int8_t, Ascending)
                         : compareFunction(Int8, int8_t, Descending);
    break;
  case ONNX_TYPE_UINT16:
    compFunc = ascending ? compareFunction(Uint16, uint16, Ascending)
                         : compareFunction(Uint16, uint16, Descending);
    break;
  case ONNX_TYPE_INT16:
    compFunc = ascending ? compareFunction(Int16, int16, Ascending)
                         : compareFunction(Int16, int16, Descending);
    break;
  case ONNX_TYPE_UINT32:
    compFunc = ascending ? compareFunction(Uint32, uint32, Ascending)
                         : compareFunction(Uint32, uint32, Descending);
    break;
  case ONNX_TYPE_INT32:
    compFunc = ascending ? compareFunction(Int32, int32, Ascending)
                         : compareFunction(Int32, int32, Descending);
    break;
  case ONNX_TYPE_UINT64:
    compFunc = ascending ? compareFunction(Uint64, uint64, Ascending)
                         : compareFunction(Uint64, uint64, Descending);
    break;
  case ONNX_TYPE_INT64:
    compFunc = ascending ? compareFunction(Int64, int64, Ascending)
                         : compareFunction(Int64, int64, Descending);
    break;
  case ONNX_TYPE_FLOAT:
    compFunc = ascending ? compareFunction(Float, float, Ascending)
                         : compareFunction(Float, float, Descending);
    break;
  case ONNX_TYPE_DOUBLE:
    compFunc = ascending ? compareFunction(Double, double, Ascending)
                         : compareFunction(Double, double, Descending);
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
#pragma warning(disable : 4028)
  // Windows supports qsort_s
  sortFunctionType *sortFunc = qsort_s;
#pragma warning(pop)
#else
  // Use standard quick sort in libc
  sortFunctionType *sortFunc = qsort_r;
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
