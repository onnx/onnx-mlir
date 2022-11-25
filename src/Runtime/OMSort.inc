#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"
#include "src/Runtime/OMSort.h"
#ifdef __cplusplus
#include "src/Runtime/OMTensorHelper.hpp"
#endif

#define TRUE 1
#define FALSE 0

// Declare comparation funcions for data types and sorting directions
typedef int (compareFuncType(void *dataPtr, uint64_t off1, uint64_t off2));
#define declare_compare_function(fname, typeName, direction, symbol) \
int compare##fname##direction(void *dataPtr, uint64_t off1, uint64_t off2) { \
  return ((typeName *)dataPtr)[off1] symbol ((typeName *)dataPtr)[off2]; \
}
#define compareFunction(fname, typeName, direction) compare##fname##direction
// clang-format off
// declare ascending functions
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
// clang-format on

// Declare sort functions for sorting algorithms
typedef void (sortFuncType(void *dataPtr, uint64_t *idx, uint64_t dataNum,
    compareFuncType compFunc));

// Bubble sort
void bubble_sort(void *dataPtr, uint64_t *idx, uint64_t dataNum,
    compareFuncType compFunc) {
  for (uint32_t i = 0; i < dataNum; i++) {
    for (uint32_t j = 0; j < dataNum - 1; j++) {
      if (compFunc(dataPtr, idx[j], idx[j + 1])) {
        int64_t tmp = idx[j];
        idx[j] = idx[j + 1];
        idx[j + 1] = tmp;
      }
    }
  }
}

// Comb sort
void comb_sort(void *dataPtr, uint64_t *idx, uint64_t dataNum,
    compareFuncType compFunc) {
  size_t gap = dataNum;
  int swap = TRUE;
  while ((gap != 1) || swap) {
    gap = gap / 1.3;
    gap = (gap < 1) ? 1 : gap;
    swap = FALSE;
    for (uint32_t i = 0; i < dataNum - gap; i++) {
      if (compFunc(dataPtr, idx[i], idx[i + gap])) {
        int64_t tmp = idx[i];
        idx[i] = idx[i + gap];
        idx[i + gap] = tmp;
        swap = TRUE;
      }
    }
  }
}

compareFuncType *getCompareFunction(uint64_t ascending, OM_DATA_TYPE dataType) {
  compareFuncType (*compFunc);

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
    uint64_t axis, uint64_t ascending, uint64_t algorithm) {
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

  // Choose the appropriate compare function
  compareFuncType (* compareElements) = getCompareFunction(ascending, dataType);

  // Choose the sorting function
  sortFuncType (* sortFunc);
  switch (algorithm) {
  case OMSORT_BUBBLE:
    sortFunc = bubble_sort;
    break;
  case OMSORT_COMB:
    sortFunc = comb_sort;
    break;
  default:
    assert("invalid sorting algorithm");
    break;
  }
  // To support input Tensor with various ranks in a uniform way.
  // If the input rank < 6, upgrade the rank to 6 virtually without changing
  // the physical memory layout by inserting length=1 ranks at lower ranks.
  // The sort axis is changed to 5 always.
  int64_t shape[6] = {1, 1, 1, 1, 1, 1};
  int64_t strides[6] = {0, 0, 0, 0, 0, 0};
  for (uint64_t i = 0; i < rank; i++) {
    shape[i + (6 - rank)] = inputShape[i];
    strides[i + (6 - rank)] = inputStrides[i];
  }

  // Sort the the 6th axis in the outer 5 loops
  int64_t sort_elems = inputShape[axis];
  for (int dim0 = 0; dim0 < shape[0]; dim0++) {
    for (int dim1 = 0; dim1 < shape[1]; dim1++) {
      for (int dim2 = 0; dim2 < shape[2]; dim2++) {
        for (int dim3 = 0; dim3 < shape[3]; dim3++) {
          for (int dim4 = 0; dim4 < shape[4]; dim4++) {
            uint64_t off = dim0 * strides[0] + dim1 * strides[1] +
                          dim2 * strides[2] + dim3 * strides[3] +
                          dim4 * strides[4];
            uint64_t *idx = order + off;
            sortFunc(dataPtr, idx, sort_elems, compareElements);
          }
        }
      }
    }
  }
  return;
}
