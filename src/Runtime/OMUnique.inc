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
  uint64_t *indicesPtr;
  uint64_t *inverseIndicesPtr;
  uint64_t *countsPtr;
  uint64_t sorted;
} uniqueTable;

void uniqueTableInit(uniqueTable *table, OM_DATA_TYPE dataType,
    uint64_t elemSize, uint64_t elemMax, void *tablePtr, uint64_t *indicesPtr,
    uint64_t *inverseIndicesPtr, uint64_t *countsPtr, uint64_t sorted) {
  table->dataType = dataType;
  table->elementSize = elemSize;
  table->elementMax = elemMax;
  table->elementCount = 0;
  table->tablePtr = tablePtr;
  table->indicesPtr = indicesPtr;
  table->inverseIndicesPtr = inverseIndicesPtr;
  table->countsPtr = countsPtr;
  table->sorted = sorted;
}

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
    if (table->sorted &&
        !isLessElem(elementInTable, element, table->elementSize,
            table->dataType)) {
      break;
    }
  }
  if (found == 0) { // no matching element found in the table
    // make space to insert an element
    for (uint64_t j = table->elementCount; j > i; j--) {
      void *elem = (void *)(tablePtr + table->elementSize * j);
      void *elemPrev = (void *)(tablePtr + table->elementSize * (j - 1));
      memcpy(elem, elemPrev, table->elementSize);
      if (table->countsPtr != NULL)
        (table->countsPtr)[j] = (table->countsPtr)[j - 1];
      if (table->indicesPtr != NULL)
        (table->indicesPtr)[j] = (table->indicesPtr)[j - 1];
    }
    if (table->inverseIndicesPtr != NULL) {
      for (uint64_t j = 0; j < off; j++) {
        if ((table->inverseIndicesPtr)[j] >= i)
          (table->inverseIndicesPtr)[j] += 1;
      }
    }
    (table->elementCount)++;
    // insert the current element to the space
    void *elementInTable = (void *)(tablePtr + table->elementSize * i);
    memcpy(elementInTable, element, table->elementSize);
    if (table->countsPtr != NULL)
      (table->countsPtr)[i] = 1;
    if (table->indicesPtr != NULL)
      (table->indicesPtr)[i] = off;
  }
  if (table->inverseIndicesPtr != NULL)
    (table->inverseIndicesPtr)[off] = i;
  return found;
}

int64_t uniqueTable_count(uniqueTable *table) { return table->elementCount; }

void getSliceData(void *inputPtr, uint64_t *shape, uint64_t *strides, uint64_t rank,
    uint64_t *shapeInUniqueAxis, uint64_t sliceAxis, uint64_t idxInSliceAxis,
    uint64_t dataSize, void *sliceData) {
  assert(rank == 6 && "rank should be expanded to 6 in getSliceData");
#if 1
printf("XXXX getSliceData: inputPtr=[%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld], "
       "shape=[%ld:%ld:%ld:%ld:%ld:%ld], "
       "strides=[%ld:%ld:%ld:%ld:%ld:%ld], rank=%ld, "
       "shapeInUniqueAxis=[%ld:%ld:%ld:%ld:%ld:%ld], "
       "sliceAxis=%ld, idxInSliceAxis=%ld, dataSize=%ld\n",
       ((uint64_t *) inputPtr)[0], ((uint64_t *) inputPtr)[1],
       ((uint64_t *) inputPtr)[2], ((uint64_t *) inputPtr)[3],
       ((uint64_t *) inputPtr)[4], ((uint64_t *) inputPtr)[5],
       ((uint64_t *) inputPtr)[6], ((uint64_t *) inputPtr)[7],
       ((uint64_t *) inputPtr)[8], ((uint64_t *) inputPtr)[9],
       ((uint64_t *) inputPtr)[10], ((uint64_t *) inputPtr)[11],
       shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
       strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
       rank, shapeInUniqueAxis[0], shapeInUniqueAxis[1], shapeInUniqueAxis[2],
       shapeInUniqueAxis[3], shapeInUniqueAxis[4], shapeInUniqueAxis[5],
       sliceAxis, idxInSliceAxis, dataSize);
#endif  
  // Gather all data in the slice
  uint64_t dim[6];
  int dataCount = 0;
  for (dim[5] = 0; dim[5] < shapeInUniqueAxis[5]; dim[5]++) {
    for (dim[4] = 0; dim[4] < shapeInUniqueAxis[4]; dim[4]++) {
      for (dim[3] = 0; dim[3] < shapeInUniqueAxis[3]; dim[3]++) {
        for (dim[2] = 0; dim[2] < shapeInUniqueAxis[2]; dim[2]++) {
          for (dim[1] = 0; dim[1] < shapeInUniqueAxis[1]; dim[1]++) {
            for (dim[0] = 0; dim[0] < shapeInUniqueAxis[0]; dim[0]++) {
              uint64_t tdim[6] = {dim[0], dim[1], dim[2], dim[3], dim[4], dim[5]};
              tdim[sliceAxis] = idxInSliceAxis;
              uint64_t currOff = 
                  tdim[0] * strides[0] + tdim[1] * strides[1] +
                  tdim[2] * strides[2] + tdim[3] * strides[3] +
                  tdim[4] * strides[4] + tdim[5] * strides[5];
              void *currInputPtr = ((char *)inputPtr) + currOff * dataSize;
#if 1
              printf("getSliceData: tdim=[%ld:%ld:%ld:%ld:%ld:%ld], currOff=%ld, "
                     "strides=[%ld:%ld:%ld:%ld:%ld:%ld] (%ld) => dataCount=%d\n", 
                     tdim[0], tdim[1], tdim[2], tdim[3], tdim[4], tdim[5], currOff,
                     strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
                     *((uint64_t *) currInputPtr), dataCount);
#endif
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

#if 0
void uniqueTableSortAndConvertToOriginalOrder(uniqueTable *table, int64_t sorted,
    int64_t sliced, int64_t *shape, int64_t rank, int64_t *strides,
    int64_t *shapeInUniqueAxis, int64_t axis, int64_t dataSize) {
  uint64_t elemSize = table->elementSize;
  uint64_t elemCount = table->elementCount;
  uint64_t axisSize = shape[axis];
  printf("XXX uniqueTableSortAndConvertToOriginalOrder called (sliced=%ld, elmCount=%ld, elemSize=%ld)\n", sliced, elemCount, elemSize);
  char *sortedTablePtr = (char *) alloca(elemCount * elemSize);
  int64_t *sortedOrder = (int64_t *) alloca(elemCount * sizeof(int64_t));
  int64_t *sortedIndicesPtr = (int64_t *) alloca(elemCount * sizeof(int64_t));
  int64_t *sortedInverseIndicesPtr = (int64_t *) alloca(axisSize * sizeof(int64_t));
  int64_t *sortedCountsPtr = (int64_t *) alloca(elemCount * sizeof(int64_t));
  if (sorted) {
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
      printf("XXXX sortedOrder[%ld] = %d\n", idx, order);
      sortedOrder[idx] = order;
    }
    // Allocate sorted tables
    // Set sorted tables
    for (uint64_t i = 0; i < elemCount; i++) {
      memcpy(sortedTablePtr + sortedOrder[i] * elemSize,
             ((char *)table->tablePtr) + i * elemSize, elemSize);
      if (table->indicesPtr != NULL) {
        sortedIndicesPtr[sortedOrder[i]] = table->indicesPtr[i];
        printf("XXXXXX sortedOrder[%ld]=%ld, table->indicesPtr[i]=%ld\n",
               i, sortedOrder[i], table->indicesPtr[i]);
      }
      if (table->countsPtr != NULL)
        sortedCountsPtr[sortedOrder[i]] = table->countsPtr[i];
    }
  } else {
    memcpy(sortedTablePtr, table->tablePtr, elemCount * elemSize);
  }
  if (table->inverseIndicesPtr != NULL)
    for (uint64_t i = 0; i < axisSize; i++)
      sortedInverseIndicesPtr[i] = sortedOrder[table->inverseIndicesPtr[i]];

  if (sliced) {
    // Copy sorted data from the sorted tables to the original tables
#if 0
    printf("XXXXXX elmCount = %ld\n", elemCount);
    memcpy(table->tablePtr, (void *)sortedTablePtr, elemCount * elemSize * dataSize);
#else
    for (int idxInSliceAxis = 0; idxInSliceAxis < (int) elemCount; idxInSliceAxis++) {
      void *slicedData = sortedTablePtr + idxInSliceAxis * elemSize;
      getSliceData(table->tablePtr, shape, strides, rank, shapeInUniqueAxis,
          axis, sortedOrder[idxInSliceAxis], dataSize, slicedData); 
    }
#endif
  }
  if (sliced && sorted) {
    if (table->countsPtr != NULL)
      memcpy(table->countsPtr, sortedCountsPtr, elemCount * sizeof(int64_t));
    if (table->indicesPtr != NULL)
      memcpy(table->indicesPtr, sortedIndicesPtr, elemCount * sizeof(int64_t));
    if (table->inverseIndicesPtr != NULL)
      memcpy(table->inverseIndicesPtr, sortedInverseIndicesPtr,
             axisSize * sizeof(int64_t));
  }
  
  return;
}

void uniqueTableSort(uniqueTable *table) {
  return uniqueTableSortAndConvertToOriginalOrder(table, /*sorted=*/1,
      /*sliced=*/0, /*shape=*/NULL, /*rank=*/0, /*strides=*/NULL, /*shapeInUniqueAxis=*/NULL,
      /*axis=*/0, /*dataSize=*/0);
}
#endif

void transposeY(OMTensor *Y, uint64_t *shape, uint64_t rank, uint64_t axis, uint64_t count) {
  assert(rank == 6 && "rank should be expanded to 6 in transposeY");
  const OM_DATA_TYPE dataType = omTensorGetDataType(Y);
  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];
  uint64_t strides[6];
  uint64_t elementNum = 1;
  uint64_t sliceSize = 1;
  for (uint64_t i = 0; i < rank; i++) {
    strides[i] = (i == 0) ? 1 : strides[i - 1] * shape[i - 1];
    elementNum *= (i == axis) ? count : shape[i];
    sliceSize *= (i == axis) ? 1 : shape[i];
  }
  void *inputPtr = alloca(dataSize * elementNum);
  memcpy(inputPtr, omTensorGetDataPtr(Y), dataSize * elementNum);
  void *outputPtr = omTensorGetDataPtr(Y);
#if 1
printf("XXXX transposeY: inputPtr=[%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld], "
       "shape=[%ld:%ld:%ld:%ld:%ld:%ld], "
       "strides=[%ld:%ld:%ld:%ld:%ld:%ld], rank=%ld, "
       "axis=%ld, dataSize=%ld\n",
       ((uint64_t *) inputPtr)[0], ((uint64_t *) inputPtr)[1],
       ((uint64_t *) inputPtr)[2], ((uint64_t *) inputPtr)[3],
       ((uint64_t *) inputPtr)[4], ((uint64_t *) inputPtr)[5],
       ((uint64_t *) inputPtr)[6], ((uint64_t *) inputPtr)[7],
       ((uint64_t *) inputPtr)[8], ((uint64_t *) inputPtr)[9],
       ((uint64_t *) inputPtr)[10], ((uint64_t *) inputPtr)[11],
       shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
       strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
       rank, axis, dataSize);
#endif  
  uint64_t tshape[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]};
  tshape[axis] = 1;
  uint64_t dim[6];
  uint64_t inputIdx = 0;
  for (uint64_t idxInSliceAxis = 0; idxInSliceAxis < count; idxInSliceAxis++) {
    uint64_t sliceOff = sliceSize * idxInSliceAxis;
    //printf("XXXX sliceSize=%ld, idxInSliceAxis=%ld, sliceOff=%ld\n", sliceSize, idxInSliceAxis, sliceOff);
    for (dim[0] = 0; dim[0] < tshape[0]; dim[0]++) {
      for (dim[1] = 0; dim[1] < tshape[1]; dim[1]++) {
        for (dim[2] = 0; dim[2] < tshape[2]; dim[2]++) {
          for (dim[3] = 0; dim[3] < tshape[3]; dim[3]++) {
            for (dim[4] = 0; dim[4] < tshape[4]; dim[4]++) {
              for (dim[5] = 0; dim[5] < tshape[5]; dim[5]++) {
                void *currInputPtr = ((char *)inputPtr) + inputIdx * dataSize;
                 uint64_t tdim[6] = {dim[0], dim[1], dim[2], dim[3], dim[4], dim[5]};
                tdim[axis] = idxInSliceAxis;
                uint64_t currOff = 
                    tdim[0] * strides[0] + tdim[1] * strides[1] +
                    tdim[2] * strides[2] + tdim[3] * strides[3] +
                    tdim[4] * strides[4] + tdim[5] * strides[5];
                void *currOutputPtr = ((char *)outputPtr) + currOff * dataSize;
#if 1
                printf("XXXX tdim=[%ld:%ld:%ld:%ld:%ld:%ld], inputIdx=%ld, currOff=%ld, dataSize=%ld: "
                       "currInput=%ld\n", tdim[0], tdim[1], tdim[2], tdim[3], tdim[4], tdim[5],
                       inputIdx, currOff, dataSize, ((uint64_t *) currInputPtr)[0]);
#endif
                memcpy(currOutputPtr, currInputPtr, dataSize);
                inputIdx++;
             }
            }
          }
        }
      }
    }
  }
}

void omTensorUnique(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t sliceAxis, uint64_t sorted, OMTensor *Y, OMTensor *indices,
    OMTensor *inverse_indices, OMTensor *counts) {
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
        (uint64_t *)indicesPtr, (uint64_t *)inverseIndicesPtr,
        (uint64_t *)countsPtr, sorted);
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
    uint64_t shape[6] = {1, 1, 1, 1, 1, 1};
    uint64_t shapeInUniqueAxis[6] = {1, 1, 1, 1, 1, 1};
    uint64_t strides[6] = {0, 0, 0, 0, 0, 0};
    uint64_t rank = 6;
    for (int64_t i = 0; i < inputRank; i++) {
      shape[i + (6 - inputRank)] = inputShape[i];
      strides[i + (6 - inputRank)] = inputStrides[i];
      shapeInUniqueAxis[i + (6 - inputRank)] = (i == sliceAxis) ? 1 : inputShape[i];
    }
    uint64_t axis = sliceAxis + (6 - inputRank);
    int64_t sliceNum = inputShape[sliceAxis];
    int64_t sliceSize = 1;
    for (int64_t i = 0; i < inputRank; i++)
      sliceSize *= (i == sliceAxis) ? 1 : inputShape[i];
    void *tablePtr = (YPtr == NULL) ? alloca(sliceSize * sliceNum * dataSize) : YPtr;
    void *sliceData = alloca(sliceSize * dataSize);
    uniqueTableInit(&uniqueTable, dataType, sliceSize * dataSize, sliceNum,
        tablePtr, (uint64_t *)indicesPtr, (uint64_t *)inverseIndicesPtr,
        (uint64_t *)countsPtr, sorted);
    for (int idxInSliceAxis = 0; idxInSliceAxis < sliceNum; idxInSliceAxis++) {
      getSliceData(inputPtr, shape, strides, rank, shapeInUniqueAxis, axis,
          idxInSliceAxis, dataSize, sliceData);
      if (uniqueTableRegister(&uniqueTable, sliceData, idxInSliceAxis) == 0) {
        count++;
      }
    }
    if ((sliceAxis > 0) && (Y != NULL)) {
      uint64_t YShape[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]};
      YShape[axis] = count;
      transposeY(Y, YShape, rank, axis, count);
    }
  }
  *totalPtr = count;
#if 0
  printf("OMUnique: return (count = %d) YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY\n", count);
  omTensorPrint("INPUT [\n", inputTensor);
  if (Y != NULL)
    omTensorPrint("] Y [\n", Y);
  if (indices != NULL)
    omTensorPrint("] INDICES [\n", indices);
  if (inverse_indices != NULL)
    omTensorPrint("] INV_INDICES [\n", inverse_indices);
  printf("] YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY\n");
#endif
  return;
}

void omTensorUniqueCount(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t sliceAxis, uint64_t sorted) {
  omTensorUnique(
      totalTensor, inputTensor, sliceAxis, sorted, NULL, NULL, NULL, NULL);
  return;
}
