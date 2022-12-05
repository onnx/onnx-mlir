#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#define __USE_GNU
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"
#include "src/Runtime/OMSort.h"
#ifdef __cplusplus
#include "src/Runtime/OMTensorHelper.hpp"
#endif

#define TRUE 1
#define FALSE 0

//
// Declare comparation funcions for data types and sorting directions
//
#ifdef __APPLE__
typedef int(
    compareFunctionType(void *dataPtr, const void *idx1, const void *idx2));
#define declare_compare_function(fname, typeName, direction, symbol)           \
  int compare##fname##direction(                                               \
      void *dataPtr, const void *idx1, const void *idx2) {                     \
    return ((typeName *)dataPtr)[*((uint64_t *)idx1)] symbol(                  \
        (typeName *)dataPtr)[*((uint64_t *)idx2)];                             \
  }
#else
typedef int(
    compareFunctionType(const void *idx1, const void *idx2, void *dataPtr));
#define declare_compare_function(fname, typeName, direction, symbol)           \
  int compare##fname##direction(                                               \
      const void *idx1, const void *idx2, void *dataPtr) {                     \
    return ((typeName *)dataPtr)[*((uint64_t *)idx1)] symbol(                  \
        (typeName *)dataPtr)[*((uint64_t *)idx2)];                             \
  }
#endif
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

//
// Declare sort functions for sorting algorithms
//
#ifdef __APPLE__
                            typedef void(sortFuncionType(void *base,
                                size_t dataNum, size_t dataSize, void *dataPtr,
                                compareFunctionType compFunc));
#else
                            typedef void(sortFuncionType(void *base,
                                size_t dataNum, size_t dataSize,
                                compareFunctionType compFunc, void *dataPtr));
#endif
#define SWAP_INDEX(a, b)                                                       \
  do {                                                                         \
    uint64_t tmp = (a);                                                        \
    (a) = (b);                                                                 \
    (b) = tmp;                                                                 \
  } while (0)

//
// Bubble sort
//
#ifdef __APPLE__
void bubble_sort(void *base, size_t dataNum, size_t dataSize, void *dataPtr,
    compareFunctionType compFunc)
#else
void bubble_sort(void *base, size_t dataNum, size_t dataSize,
    compareFunctionType compFunc, void *dataPtr)
#endif
{
  uint64_t *idx = (uint64_t *)base;
  for (int i = 0; i < dataNum; i++) {
    for (int j = 0; j < dataNum - 1; j++) {
#ifdef __APPLE__
      if (compFunc(dataPtr, idx + j, idx + j + 1))
#else
      if (compFunc(idx + j, idx + j + 1, dataPtr))
#endif
      {
        SWAP_INDEX(idx[j], idx[j + 1]);
      }
    }
  }
}

//
// Comb sort
//
#ifdef __APPLE__
void comb_sort(void *base, size_t dataNum, size_t dataSize, void *dataPtr,
    compareFunctionType compFunc)
#else
void comb_sort(void *base, size_t dataNum, size_t dataSize,
    compareFunctionType compFunc, void *dataPtr)
#endif
{
  uint64_t *idx = (uint64_t *)base;
  size_t gap = dataNum;
  int swap = TRUE;
  while ((gap != 1) || swap) {
    gap = gap / 1.3;
    gap = (gap < 1) ? 1 : gap;
    swap = FALSE;
    for (int i = 0; i < dataNum - gap; i++) {
#ifdef __APPLE__
      if (compFunc(dataPtr, idx + i, idx + i + gap, dataPtr))
#else
      if (compFunc(idx + i, idx + i + gap, dataPtr))
#endif
      {
        SWAP_INDEX(idx[i], idx[i + gap]);
      }
    }
  }
}

//
// Quick sort
//
// stack for index (uint64_t)
typedef struct indexStack {
  uint64_t *stackData;
  int64_t stackSize;
  int64_t stackTop;
} indexStack;

#define STACK_INIT(stack, stackSize)                                           \
  do {                                                                         \
    assert(stackSize > 0);                                                     \
    stack.stackData = (uint64_t *)alloca(stackSize * sizeof(uint64_t));        \
    assert(stack.stackData != NULL);                                           \
    stack.stackSize = stackSize;                                               \
    stack.stackTop = 0;                                                        \
  } while (0)
#define STACK_ISEMPTY(stack) (stack.stackTop == 0)
#define STACK_PUSH(stack, begin, end)                                          \
  do {                                                                         \
    assert(stack.stackTop <= stack.stackSize - 2);                             \
    stack.stackData[(stack.stackTop)++] = begin;                               \
    stack.stackData[(stack.stackTop)++] = end;                                 \
  } while (0)
#define STACK_POP(stack, begin, end)                                           \
  do {                                                                         \
    assert(stack.stackTop >= 2);                                               \
    end = stack.stackData[--(stack.stackTop)];                                 \
    begin = stack.stackData[--(stack.stackTop)];                               \
  } while (0)
#define STACK_PRINT(stack)                                                     \
  do {                                                                         \
    assert(stack.stackTop >= 0);                                               \
    fprintf(stderr, "Stack: [");                                               \
    for (int64_t i = 0; (i + 1) < stack.stackTop; i += 2) {                    \
      fprintf(                                                                 \
          stderr, "<%ld:%ld>, ", stack.stackData[i], stack.stackData[i + 1]);  \
    }                                                                          \
    fprintf(                                                                   \
        stderr, "] (Top=%ld,Size=%ld)\n", stack.stackTop, stack.stackSize);    \
    fflush(stderr);                                                            \
  } while (0)

static int64_t log2u(uint64_t n) {
  assert(n > 0);
  int64_t b = 0;
  for (; n > 0; b++)
    n = n >> 2;
  return b;
}

// Quick sort patition function
static int64_t quick_sort_partition(void *dataPtr, uint64_t *idx,
    compareFunctionType compFunc, int64_t begin, int64_t end) {
  int64_t i = begin;
  for (int64_t j = begin; j < end; j++) {
#ifdef __APPLE__
    if (!compFunc(dataPtr, idx + j, idx + end))
#else
    if (!compFunc(idx + j, idx + end, dataPtr))
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

uint64_t getDataSize(OM_DATA_TYPE dataType) {
  uint64_t size = 0;
  switch (dataType) {
  case ONNX_TYPE_BOOL:
    size = sizeof(bool);
    break;
  case ONNX_TYPE_UINT8:
    size = sizeof(uint8_t);
    break;
  case ONNX_TYPE_INT8:
    size = sizeof(int8_t);
    break;
  case ONNX_TYPE_UINT16:
    size = sizeof(uint16_t);
    break;
  case ONNX_TYPE_INT16:
    size = sizeof(int16_t);
    break;
  case ONNX_TYPE_UINT32:
    size = sizeof(uint32_t);
    break;
  case ONNX_TYPE_INT32:
    size = sizeof(int32_t);
    break;
  case ONNX_TYPE_UINT64:
    size = sizeof(uint64_t);
    break;
  case ONNX_TYPE_INT64:
    size = sizeof(int64_t);
    break;
  case ONNX_TYPE_FLOAT:
    size = sizeof(float);
    break;
  case ONNX_TYPE_DOUBLE:
    size = sizeof(double);
    break;
  default:
    assert(false && "unexpected data type in getDataSize");
  }
  return size;
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
  int64_t sort_elems = inputShape[axis];
  // Sorting not necessary for empty array
  if (sort_elems == 0)
    return;

  // Choose the appropriate compare function
  compareFunctionType *compareElements =
      getCompareFunction(ascending, dataType);
  uint64_t datasize = getDataSize(dataType);

  // Choose the sorting function
  sortFuncionType *sortFunc = bubble_sort;
  switch (algorithm) {
  case OMSORT_BUBBLESORT:
    sortFunc = bubble_sort;
    break;
  case OMSORT_COMBSORT:
    sortFunc = comb_sort;
    break;
  case OMSORT_QUICKSORT:
#if defined(__APPLE__)
    // MacOS supports qsort_r in different API
    sortFunc = qsort_r;
#elif defined(_MSC_VER)
    // Windows supports qsort_s
    sortFunc = qsort_s;
#else
    // Use standard quick sort in libc
    sortFunc = qsort_r;
#endif
    break;
  case OMSORT_QUICKSORT_CUSTOM:
    sortFunc = quick_sort_custom; // custum quick sort
    break;
  default:
    assert("invalid sorting algorithm");
    break;
  }
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
