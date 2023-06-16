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

//
// Table for elements
//
typedef struct uniqueTable {
  OM_DATA_TYPE dataType;
  uint64_t elementSize;
  uint64_t elementMax;
  uint64_t elementCount;
  void *tablePtr;
  int64_t *indicesPtr;
  int64_t *inverseIndicesPtr;
  int64_t *countsPtr;
} uniqueTable;

void uniqueTableInit(uniqueTable *table, OM_DATA_TYPE dataType,
    uint64_t elemSize, uint64_t elemMax, void *tablePtr, int64_t *indicesPtr,
    int64_t *inverseIndicesPtr, int64_t *countsPtr) {
  table->dataType = dataType;
  table->elementSize = elemSize;
  table->elementMax = elemMax;
  table->elementCount = 0;
  table->tablePtr = tablePtr;
  table->indicesPtr = indicesPtr;
  table->inverseIndicesPtr = inverseIndicesPtr;
  table->countsPtr = countsPtr;
}

int uniqueTableRegister(uniqueTable *table, void *element, uint64_t off) {
  char *tablePtr = (char *)table->tablePtr;
  uint64_t i;
  int found = 0;
  // Searching for matching data in linear search
  // More optimizations, such as introducing hash table, are considerable.
  for (i = 0; i < table->elementCount; i++) {
    void *elementInTable = (void *)(tablePtr + table->elementSize * i);
    if (!memcmp(elementInTable, element, table->elementSize)) {
      found = 1;
      if (table->countsPtr != NULL)
        ((table->countsPtr)[i])++;
      break;
    }
  }
  if (i == table->elementCount) { // no matching element found in the table
    void *elementInTable = (void *)(tablePtr + table->elementSize * i);
    memcpy(elementInTable, element, table->elementSize);
    if (table->countsPtr != NULL)
      (table->countsPtr)[table->elementCount] = 1;
    if (table->indicesPtr != NULL)
      (table->indicesPtr)[table->elementCount] = off;
    (table->elementCount)++;
  }
  if (table->inverseIndicesPtr != NULL)
    (table->inverseIndicesPtr)[off] = i;
  return found;
}

int64_t uniqueTable_count(uniqueTable *table) { return table->elementCount; }

int isLessNum(void *arg1, void *arg2, OM_DATA_TYPE dataType) {
  switch (dataType) {
  case ONNX_TYPE_FLOAT:
    return *((float *)arg1) < *((float *)arg2);
  case ONNX_TYPE_UINT8:
    return *((uint8_t *)arg1) < *((uint8_t *)arg2);
  case ONNX_TYPE_INT8:
    return *((int8_t *)arg1) < *((int8_t *)arg2);
  case ONNX_TYPE_UINT16:
    return *((uint16_t *)arg1) < *((uint16_t *)arg2);
  case ONNX_TYPE_INT16:
    return *((int16_t *)arg1) < *((int16_t *)arg2);
  case ONNX_TYPE_INT32:
    return *((int32_t *)arg1) < *((int32_t *)arg2);
  case ONNX_TYPE_INT64:
    return *((int64_t *)arg1) < *((int64_t *)arg2);
  //case ONNX_TYPE_STRING:
  case ONNX_TYPE_BOOL:
    return *((bool *)arg1) < *((bool *)arg2);
  //case ONNX_TYPE_FLOAT16:
  case ONNX_TYPE_DOUBLE:
    return *((double *)arg1) < *((double *)arg2);
  case ONNX_TYPE_UINT32:
    return *((uint32_t *)arg1) < *((uint32_t *)arg2);
  case ONNX_TYPE_UINT64:
    return *((uint64_t *)arg1) < *((uint64_t *)arg2);
  //case ONNX_TYPE_COMPLEX64:
  //case ONNX_TYPE_COMPLEX128:
  default:
    assert("Unsupported ONNX type in OMTensor");
  }
  return 0;
}

int isLessElem(void *elem1, void *elem2, uint64_t elemSize,
    OM_DATA_TYPE dataType) {
  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];
  int64_t elemNum = elemSize / dataSize;
  for (int i = 0; i < elemNum; i++) {
    void *num1 = ((char *) elem1) + (dataSize * i);
    void *num2 = ((char *) elem2) + (dataSize * i);
    if (memcmp(num1, num2, dataSize))
      return isLessNum(num1, num2, dataType);
  }
  return 0;
}

void uniqueTableSort(uniqueTable *table) {
  uint64_t elemSize = table->elementSize;
  uint64_t elemCount = table->elementCount;
  int64_t *sortedOrder = (int64_t *) alloca(elemCount * sizeof(int64_t));
  //printf("XXXX uniqueTableSort: called\n");
  for (uint64_t idx = 0; idx < elemCount; idx++) {
    // get the least value in the table
    int order = 0;
    void *currIdxElemPtr = (void *)(((char *)table->tablePtr) + elemSize * idx);
    for (uint64_t i = 0; i < elemCount; i++) {
      void *elemPtr = (void *)(((char *)table->tablePtr) + elemSize * i);
      if (isLessElem(elemPtr, currIdxElemPtr, elemSize, table->dataType)) {
        order++;
      }
    }
    //printf("XXXX sortedOrder[%d] = %ld\n", idx, order);
    sortedOrder[idx] = order;
  }
  // Allocate sorted tables
  char *sortedTablePtr = (char *) alloca(elemCount * elemSize);
  int64_t *sortedIndicesPtr = (int64_t *) alloca(elemCount * sizeof(int64_t));
  int64_t *sortedInverseIndicesPtr = (int64_t *) alloca(elemCount * sizeof(int64_t));
  int64_t *sortedCountsPtr = (int64_t *) alloca(elemCount * sizeof(int64_t));
  // Set sorted tables
  for (uint64_t i = 0; i < elemCount; i++) {
    memcpy(sortedTablePtr + sortedOrder[i] * elemSize,
           ((char *)table->tablePtr) + i * elemSize, elemSize);
    if (table->countsPtr != NULL)
      sortedCountsPtr[sortedOrder[i]] = table->countsPtr[i];
    if (table->indicesPtr != NULL)
      sortedIndicesPtr[sortedOrder[i]] = table->indicesPtr[i];
  }
  if (table->inverseIndicesPtr != NULL)
    for (uint64_t i = 0; i < elemCount; i++)
      sortedInverseIndicesPtr[i] = sortedOrder[table->inverseIndicesPtr[i]];
  // Copy sorted data from the sorted tables to the original tables
  memcpy(table->tablePtr, (void *)sortedTablePtr, elemCount * elemSize);
  if (table->countsPtr != NULL)
    memcpy(table->countsPtr, sortedCountsPtr, elemCount * sizeof(int64_t));
  if (table->indicesPtr != NULL)
    memcpy(table->indicesPtr, sortedIndicesPtr, elemCount * sizeof(int64_t));
  if (table->inverseIndicesPtr != NULL)
    memcpy(table->inverseIndicesPtr, sortedInverseIndicesPtr, elemCount * sizeof(int64_t));
}

void getSliceData(void *inputPtr, int64_t *shape, int64_t *strides, int64_t rank,
    int64_t *shapeInUniqueAxis, int64_t sliceAxis, int64_t idxInSliceAxis,
    int64_t sliceSize, int64_t dataSize, void *sliceData) {
  assert(rank == 6 && "rank should be expanded to 6 in getSliceData");
#if 0
printf("XXXX getSliceData: inputPtr=[%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld], "
       "shape=[%ld:%ld:%ld:%ld:%ld:%ld], "
       "strides=[%ld:%ld:%ld:%ld:%ld:%ld], rank=%ld, "
       "shapeInUniqueAxis=[%ld:%ld:%ld:%ld:%ld:%ld], "
       "sliceAxis=%ld, idxInSliceAxis=%ld, sliceSize=%ld, dataSize=%ld\n",
       ((uint64_t *) inputPtr)[0], ((uint64_t *) inputPtr)[1],
       ((uint64_t *) inputPtr)[2], ((uint64_t *) inputPtr)[3],
       ((uint64_t *) inputPtr)[4], ((uint64_t *) inputPtr)[5],
       ((uint64_t *) inputPtr)[6], ((uint64_t *) inputPtr)[7],
       ((uint64_t *) inputPtr)[8],
       shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
       strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
       rank, shapeInUniqueAxis[0], shapeInUniqueAxis[1], shapeInUniqueAxis[2],
       shapeInUniqueAxis[3], shapeInUniqueAxis[4], shapeInUniqueAxis[5],
       sliceAxis, idxInSliceAxis, sliceSize, dataSize);
#endif  
  // Gather all data in the slice
  int dim[6];
  int dataCount = 0;
  for (dim[0] = 0; dim[0] < shapeInUniqueAxis[0]; dim[0]++) {
    for (dim[1] = 0; dim[1] < shapeInUniqueAxis[1]; dim[1]++) {
      for (dim[2] = 0; dim[2] < shapeInUniqueAxis[2]; dim[2]++) {
        for (dim[3] = 0; dim[3] < shapeInUniqueAxis[3]; dim[3]++) {
          for (dim[4] = 0; dim[4] < shapeInUniqueAxis[4]; dim[4]++) {
            for (dim[5] = 0; dim[5] < shapeInUniqueAxis[5]; dim[5]++) {
              int tdim[6];
              for (int i = 0; i < 6; i++)
                tdim[i] = dim[i];
              tdim[sliceAxis] = idxInSliceAxis;
              uint64_t currOff = 
                  tdim[0] * strides[0] + tdim[1] * strides[1] +
                  tdim[2] * strides[2] + tdim[3] * strides[3] +
                  tdim[4] * strides[4] + tdim[5] * strides[5];
              void *currInputPtr = ((char *)inputPtr) + currOff * dataSize;
              void *currSliceData = ((char *)sliceData) + dataCount * dataSize;
              memcpy(currSliceData, currInputPtr, dataSize);
              dataCount++;
            }
          }
        }
      }
    }
  }
#if 1
  printf("XXXX getSliceData returning: [");
  for (int i = 0; i < dataCount; i++) {
    printf("%ld,", ((uint64_t *) sliceData)[i]);
  }
  printf("] =============================================================\n");
#endif
}

void omTensorUnique(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t sliceAxis, uint64_t sorted, OMTensor *Y, OMTensor *indices,
    OMTensor *inverse_indices, OMTensor *counts) {
  printf("ZZZZZ: sliceAxis=%ld, sorted=%ld\n", sliceAxis, sorted);
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  const int64_t inputRank = omTensorGetRank(inputTensor);
  assert(inputRank <= 6 && "omTensorUnique assumes inputRank <= 6");
  int64_t *totalPtr = (int64_t *)omTensorGetDataPtr(totalTensor);
  const int64_t *inputShape = omTensorGetShape(inputTensor);
  const int64_t *inputStrides = omTensorGetStrides(inputTensor);  
  void *inputPtr = omTensorGetDataPtr(inputTensor);
  void *YPtr = (Y != NULL) ? omTensorGetDataPtr(Y) : NULL;
  void *indicesPtr = (indices != NULL) ? omTensorGetDataPtr(indices) : NULL;
  void *inverseIndicesPtr =
      (inverse_indices != NULL) ? omTensorGetDataPtr(inverse_indices) : NULL;
  void *countsPtr = (counts != NULL) ? omTensorGetDataPtr(counts) : NULL;
  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];

  int count = 0;
  uniqueTable uniqueTable;
  if (sliceAxis < 0) { // if slice attribute is not specified
    // manage the inputTensor as flatten one
    uint64_t elementNum = 1;
    for (int64_t i = 0; i < inputRank; i++) {
      elementNum *= inputShape[i];
    }
    void *tablePtr = (YPtr == NULL) ? alloca(dataSize * elementNum) : YPtr;
    uniqueTableInit(&uniqueTable, dataType, dataSize, elementNum, tablePtr,
        (int64_t *)indicesPtr, (int64_t *)inverseIndicesPtr,
        (int64_t *)countsPtr);
    for (uint64_t off = 0; off < elementNum; off++) {
      void *elemPtr = ((char *)inputPtr) + dataSize * off;
      if (uniqueTableRegister(&uniqueTable, elemPtr, off) == 0) {
        count++;
      }
    }
  } else { // if slice attribute is specified
    // To support input Tensor with various ranks in a uniform way.
    // If the input rank < 6, upgrade the rank to 6 virtually without changing
    // the physical memory layout by inserting length=1 ranks at lower ranks.
    // The 5th axis becomes the unique axis.
    int64_t shape[6] = {1, 1, 1, 1, 1, 1};
    int64_t shapeInUniqueAxis[6] = {1, 1, 1, 1, 1, 1};
    int64_t strides[6] = {0, 0, 0, 0, 0, 0};
    int64_t rank = 6;
    int64_t axis = sliceAxis + (rank - inputRank);
    for (int64_t i = 0; i < inputRank; i++) {
      shape[i + (rank - inputRank)] = inputShape[i];
      shapeInUniqueAxis[i + (rank - inputRank)] =
          (i == sliceAxis) ? 1 : inputShape[i];
      strides[i + (rank - inputRank)] = inputStrides[i];
    }
    int64_t sliceNum = inputShape[sliceAxis];
    int64_t sliceSize = 1;
    for (int64_t i = 0; i < inputRank; i++)
      sliceSize *= (i == sliceAxis) ? 1 : inputShape[i];
    printf("XXXXX shape=[%ld:%ld:%ld:%ld:%ld:%ld], ", shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
    printf("sliceAxis=%ld, sliceNum=%ld, sliceSize=%ld, dataSize=%ld\n", sliceAxis, sliceNum, sliceSize, dataSize);
    void *tablePtr = (YPtr == NULL) ? alloca(sliceSize * sliceNum * dataSize) : YPtr;
    uniqueTableInit(&uniqueTable, dataType, sliceSize * dataSize, sliceNum,
        tablePtr, (int64_t *)indicesPtr, (int64_t *)inverseIndicesPtr,
        (int64_t *)countsPtr);
    void *sliceData = alloca(sliceSize * dataSize);
    for (int idxInSliceAxis = 0; idxInSliceAxis < sliceNum; idxInSliceAxis++) {
      printf("XXXX idxInSliceAxis=%d\n", idxInSliceAxis);
      getSliceData(inputPtr, shape, strides, rank, shapeInUniqueAxis, axis,
          idxInSliceAxis, sliceSize, dataSize, sliceData);
      if (uniqueTableRegister(&uniqueTable, sliceData, idxInSliceAxis) == 0) {
        count++;
      }
    }
  }
  *totalPtr = count;
  printf("XXXX OMUnique: count = %d\n", count);
  if ((sorted > 0) && (Y != NULL)) {
    uniqueTableSort(&uniqueTable);
  }
  return;
}

void omTensorUniqueCount(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t sliceAxis, uint64_t sorted) {
  omTensorUnique(
      totalTensor, inputTensor, sliceAxis, sorted, NULL, NULL, NULL, NULL);
  return;
}
