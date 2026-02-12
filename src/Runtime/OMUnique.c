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
// Table for managing sliced elements
//
typedef struct sliceTable {
  OM_DATA_TYPE dataType;
  uint64_t sorted;
  uint64_t numberOfElementsInSlice;
  uint64_t numberOfSlices;
  uint64_t maxNumberOfSlices;
  void *sliceDataPtr;
  uint64_t *indicesPtr;
  uint64_t *inverseIndicesPtr;
  uint64_t *countsPtr;
} sliceTable;

void sliceTableInit(sliceTable *table, OM_DATA_TYPE dataType, uint64_t sorted,
    uint64_t numberOfElementsInSlice, uint64_t maxNumberOfSlices,
    void *sliceDataPtr, uint64_t *indicesPtr, uint64_t *inverseIndicesPtr,
    uint64_t *countsPtr) {
  table->dataType = dataType;
  table->sorted = sorted;
  table->numberOfElementsInSlice = numberOfElementsInSlice;
  table->numberOfSlices = 0;
  table->maxNumberOfSlices = maxNumberOfSlices;
  table->sliceDataPtr = sliceDataPtr;
  table->indicesPtr = indicesPtr;
  table->inverseIndicesPtr = inverseIndicesPtr;
  table->countsPtr = countsPtr;
}

#ifdef _DEBUG_SLICE_TABLE
void sliceTablePrint(sliceTable *table) {
  printf(
      "sliceTablePrint: dataType=%d, sorted=%ld, numberOfElementsInSlice=%ld, "
      "numberOfSlices=%ld, maxNumberOfSlices=%ld, sliceTable = [\n",
      table->dataType, table->sorted, table->numberOfElementsInSlice,
      table->maxNumberOfSlices, table->numberOfSlices);
  for (uint64_t i = 0; i < table->numberOfSlices; i++) {
    printf("  %ld:[", i);
    for (uint64_t j = 0; j < table->numberOfElementsInSlice; j++) {
      switch (table->dataType) {
      case ONNX_TYPE_INT64:
      case ONNX_TYPE_UINT64:
        printf("%ld,", ((int64_t *)(table->sliceDataPtr))
                           [i * table->numberOfElementsInSlice + j]);
        break;
      case ONNX_TYPE_FLOAT:
        printf("%f,", ((float *)(table->sliceDataPtr))
                          [i * table->numberOfElementsInSlice + j]);
        break;
      default:
        printf("XX, ");
      }
    }
    printf("]\n");
  }
  printf("]\n");
}
#endif

// Check if the first argument is less than the second one
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
  // case ONNX_TYPE_STRING:
  case ONNX_TYPE_BOOL:
    return *((bool *)arg1) < *((bool *)arg2);
  // case ONNX_TYPE_FLOAT16:
  case ONNX_TYPE_DOUBLE:
    return *((double *)arg1) < *((double *)arg2);
  case ONNX_TYPE_UINT32:
    return *((uint32_t *)arg1) < *((uint32_t *)arg2);
  case ONNX_TYPE_UINT64:
    return *((uint64_t *)arg1) < *((uint64_t *)arg2);
  // case ONNX_TYPE_COMPLEX64:
  // case ONNX_TYPE_COMPLEX128:
  default:
    assert(false && "Unsupported ONNX type in OMTensor");
  }
  return 0;
}

// Check if
int isLessSlice(
    void *elem1, void *elem2, uint64_t elemSize, OM_DATA_TYPE dataType) {
  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];
  int64_t elemNum = elemSize / dataSize;
  for (int i = 0; i < elemNum; i++) {
    void *num1 = ((char *)elem1) + (dataSize * i);
    void *num2 = ((char *)elem2) + (dataSize * i);
    if (memcmp(num1, num2, dataSize)) // if num1 != num2
      return isLessNum(num1, num2, dataType);
  }
  return 0;
}

//
// If the axis attr is given in onnx.Unique, it should compare slices
// according to the specified axis. "getSliceData" is called to get a slice
// according to the given axis if the axis attr is given. This function gets
// slice data according to the specified slice axis(sliceAxis), and fills them
// to the specified pointer. Slice data's shape is the following.
// - Input data: [x, y, z] (asumming sliceAxis= 1, the "y" axis is the slice.)
// - Sliced data: [x, z, count] (s.t. count is number of unique slice shape)
//
void getSliceData(const OMTensor *inputTensor, int64_t sliceAxis,
    int64_t idxInSliceAxis, void *sliceData) {
  const int64_t inputRank = omTensorGetRank(inputTensor);
  const int64_t *inputShape = omTensorGetShape(inputTensor);
  const int64_t *inputStrides = omTensorGetStrides(inputTensor);
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  void *inputPtr = omTensorGetDataPtr(inputTensor);
  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];
  assert(inputRank <= 6 && "rank should be 6 or less");
  assert(sliceAxis < inputRank && "rank should be less than rank");

  // To support input Tensor with various ranks in a uniform way.
  // If the input rank < 6, upgrade the rank to 6 virtually without changing
  // the physical memory layout by inserting length=1 ranks at lower ranks.
  uint64_t shapeInUniqueAxis[6] = {1, 1, 1, 1, 1, 1};
  uint64_t strides[6] = {0, 0, 0, 0, 0, 0};
  for (int64_t i = 0; i < inputRank; i++) {
    shapeInUniqueAxis[i] = (i == sliceAxis) ? 1 : inputShape[i];
    strides[i] = inputStrides[i];
  }
  // Gather all data in the slice
  uint64_t dim[6];
  int64_t outputIdx = 0;

  for (dim[5] = 0; dim[5] < shapeInUniqueAxis[5]; dim[5]++) {
    for (dim[4] = 0; dim[4] < shapeInUniqueAxis[4]; dim[4]++) {
      for (dim[3] = 0; dim[3] < shapeInUniqueAxis[3]; dim[3]++) {
        for (dim[2] = 0; dim[2] < shapeInUniqueAxis[2]; dim[2]++) {
          for (dim[1] = 0; dim[1] < shapeInUniqueAxis[1]; dim[1]++) {
            for (dim[0] = 0; dim[0] < shapeInUniqueAxis[0]; dim[0]++) {
              uint64_t tdim[6] = {
                  dim[0], dim[1], dim[2], dim[3], dim[4], dim[5]};
              tdim[sliceAxis] = idxInSliceAxis;
              // calculate the input and output pointers
              uint64_t off = tdim[0] * strides[0] + tdim[1] * strides[1] +
                             tdim[2] * strides[2] + tdim[3] * strides[3] +
                             tdim[4] * strides[4] + tdim[5] * strides[5];
              void *currInputPtr = ((char *)inputPtr) + off * dataSize;
              void *outputPtr = ((char *)sliceData) + outputIdx * dataSize;
              memcpy(outputPtr, currInputPtr, dataSize);
              outputIdx++;
            }
          }
        }
      }
    }
  }
}

//
// "sliceTableRegister" registers (1) a given slice returned by getSliceData
// if// the axis attr is given or (2) a single element if the axis atter is not
// given.  It compares the current slice with all of the registered slices to
// investigate an identical slice is registered or not, and returns a bool if
// it is found or not.  If the sorted attr is given, this function keeps
// slices in sorted order. Otherwise it keeps them in the coming order.
// This function also generates outputs of onnx.Unique, such as indices,
// inverse_indeces and counts. However it does not generate Y if the axis
// attr is specified, since this function simply keeps registered slices in
// the table. The output Y is generated from input Tensor, indices and axis
// in another function.
//
int sliceTableRegister(sliceTable *table, void *slice, uint64_t off) {
  uint64_t dataSize = OM_DATA_TYPE_SIZE[table->dataType];
  char *sliceDataPtr = (char *)table->sliceDataPtr;
  uint64_t sliceSizeInBytes = table->numberOfElementsInSlice * dataSize;
  // Searching for matching data in linear search
  // More optimizations, such as introducing hash table, are considerable.
  uint64_t insertIdx;
  int found = 0;
  for (insertIdx = 0; insertIdx < table->numberOfSlices; insertIdx++) {
    void *sliceInTable = (void *)(sliceDataPtr + sliceSizeInBytes * insertIdx);
    if (!memcmp(sliceInTable, slice, sliceSizeInBytes)) {
      found = 1;
      if (table->countsPtr != NULL)
        ((table->countsPtr)[insertIdx])++;
      break;
    }
    if (table->sorted &&
        !isLessSlice(sliceInTable, slice, sliceSizeInBytes, table->dataType)) {
      break;
    }
  }
  if (found == 0) { // no matching slice found in the table
    // make space to insert an slice at insertIdx
    for (uint64_t j = table->numberOfSlices; j > insertIdx; j--) {
      void *currSlice = (void *)(sliceDataPtr + sliceSizeInBytes * j);
      void *prevSlice = (void *)(sliceDataPtr + sliceSizeInBytes * (j - 1));
      memcpy(currSlice, prevSlice, sliceSizeInBytes);
      if (table->countsPtr != NULL)
        (table->countsPtr)[j] = (table->countsPtr)[j - 1];
      if (table->indicesPtr != NULL)
        (table->indicesPtr)[j] = (table->indicesPtr)[j - 1];
    }
    if (table->inverseIndicesPtr != NULL) {
      for (uint64_t j = 0; j < off; j++) {
        if ((table->inverseIndicesPtr)[j] >= insertIdx)
          (table->inverseIndicesPtr)[j] += 1;
      }
    }
    (table->numberOfSlices)++;
    // insert the current slice to the space
    void *sliceInTable = (void *)(sliceDataPtr + sliceSizeInBytes * insertIdx);
    memcpy(sliceInTable, slice, sliceSizeInBytes);
    if (table->countsPtr != NULL)
      (table->countsPtr)[insertIdx] = 1;
    if (table->indicesPtr != NULL)
      (table->indicesPtr)[insertIdx] = off;
  }
  if (table->inverseIndicesPtr != NULL)
    (table->inverseIndicesPtr)[off] = insertIdx;
  return found;
}

//
// "produceY" produces output Y from input tensor, indices and sliceAxis
// if the axis attr is given and output Y is required.
// If the axis attr is not given, sliceTableRegister generates the output Y
//  directly, since the output is a simple array of unique elements.
//
void produceY(const OMTensor *inputTensor, OMTensor *indices, int64_t sliceAxis,
    OMTensor *Y) {
  const int64_t inputRank = omTensorGetRank(inputTensor);
  assert(inputRank <= 6 && "input rank should be 6 or less");
  assert(sliceAxis < inputRank && "sliceAxis should be less than input rank");
  const int64_t *inputShape = omTensorGetShape(inputTensor);
  const int64_t *inputStrides = omTensorGetStrides(inputTensor);
  void *inputPtr = omTensorGetDataPtr(inputTensor);
  const int64_t indicesRank = omTensorGetRank(indices);
  assert(indicesRank == 1 && "indices rank should be 1");
  const int64_t *indicesShape = omTensorGetShape(indices);
  const int64_t count = indicesShape[0];
  int64_t *indicesPtr = (int64_t *)omTensorGetDataPtr(indices);
  void *YPtr = omTensorGetDataPtr(Y);

  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];

  // To support input Tensor with various ranks in a uniform way.
  // If the input rank < 6, upgrade the rank to 6 virtually without changing
  // the physical memory layout by inserting length=1 ranks at lower ranks.
  // new axis becomes
  int64_t shape[6] = {1, 1, 1, 1, 1, 1};
  int64_t outShape[6] = {1, 1, 1, 1, 1, 1};
  int64_t strides[6] = {1, 1, 1, 1, 1, 1};
  int64_t outStrides[6] = {1, 1, 1, 1, 1, 1};
  for (int64_t i = 0; i < inputRank; i++) {
    shape[i] = inputShape[i];
    outShape[i] = (i == sliceAxis) ? count : inputShape[i];
    strides[i] =
        inputStrides[i]; //(i == 0) ? 1 : (strides[i - 1] * inputShape[i - 1]);
  }
  for (int64_t i = inputRank - 1; i > 0; i--) {
    outStrides[i - 1] = outStrides[i] * outShape[i];
  }
  shape[sliceAxis] = count;
  int64_t dim[6];
  for (dim[5] = 0; dim[5] < shape[5]; dim[5]++) {
    for (dim[4] = 0; dim[4] < shape[4]; dim[4]++) {
      for (dim[3] = 0; dim[3] < shape[3]; dim[3]++) {
        for (dim[2] = 0; dim[2] < shape[2]; dim[2]++) {
          for (dim[1] = 0; dim[1] < shape[1]; dim[1]++) {
            for (dim[0] = 0; dim[0] < shape[0]; dim[0]++) {
              int64_t tdim[6] = {
                  dim[0], dim[1], dim[2], dim[3], dim[4], dim[5]};
              tdim[sliceAxis] = indicesPtr[dim[sliceAxis]];
              // calculate the input offset and pointer
              uint64_t inputOff = tdim[0] * strides[0] + tdim[1] * strides[1] +
                                  tdim[2] * strides[2] + tdim[3] * strides[3] +
                                  tdim[4] * strides[4] + tdim[5] * strides[5];
              void *inputNumPtr = ((char *)inputPtr) + inputOff * dataSize;
              // calculate the output offset and pointer
              uint64_t outputOff =
                  dim[0] * outStrides[0] + dim[1] * outStrides[1] +
                  dim[2] * outStrides[2] + dim[3] * outStrides[3] +
                  dim[4] * outStrides[4] + dim[5] * outStrides[5];
              void *YNumPtr = ((char *)YPtr) + outputOff * dataSize;
              memcpy(YNumPtr, inputNumPtr, dataSize);
            }
          }
        }
      }
    }
  }
}

//
// "omTensorUnique" handles two cases, where are (1) case with no axis given
// (2) case with axis given.
// -  case with no axis given
// In this case, omTensorUnique gets one element in the input Tensor in turn,
// calls sliceTableRegister using an element as a slice. The registered
// elements are used as the output Y.
// - case with axis given
// In this case, omTensorUnique gets a slice with getSliceData, registers it
// with getSliceData, The outputs indices, inverse_indices and counts are set
// by getSliceData. If the output Y is necessary, it is created by produceY.
//
void omTensorUnique(OMTensor *totalTensor, OMTensor *Y, OMTensor *indices,
    OMTensor *inverse_indices, OMTensor *counts, const OMTensor *inputTensor,
    int64_t sliceAxis, uint64_t sorted) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  const int64_t inputRank = omTensorGetRank(inputTensor);
  assert(inputRank <= 6 && "input rank should be 6 or less");
  assert(sliceAxis < inputRank && "axis should be less than rank");
  int64_t *totalPtr = (int64_t *)omTensorGetDataPtr(totalTensor);
  const int64_t *inputShape = omTensorGetShape(inputTensor);
  void *inputPtr = omTensorGetDataPtr(inputTensor);
  void *YPtr = (Y != NULL) ? omTensorGetDataPtr(Y) : NULL;
  void *indicesPtr = (indices != NULL) ? omTensorGetDataPtr(indices) : NULL;
  void *inverseIndicesPtr =
      (inverse_indices != NULL) ? omTensorGetDataPtr(inverse_indices) : NULL;
  void *countsPtr = (counts != NULL) ? omTensorGetDataPtr(counts) : NULL;
  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];

  int count = 0;
  sliceTable sliceTable;
  if (sliceAxis < 0) { // if slice attribute is not specified
    // manage the inputTensor as flatten one
    uint64_t elementNum = 1;
    for (int64_t i = 0; i < inputRank; i++)
      elementNum *= inputShape[i];
    void *sliceDataPtr = (YPtr == NULL) ? alloca(dataSize * elementNum) : YPtr;
    sliceTableInit(&sliceTable, dataType, sorted, 1, elementNum, sliceDataPtr,
        (uint64_t *)indicesPtr, (uint64_t *)inverseIndicesPtr,
        (uint64_t *)countsPtr);
    for (uint64_t off = 0; off < elementNum; off++) {
      void *elemPtr = ((char *)inputPtr) + dataSize * off;
      if (sliceTableRegister(&sliceTable, elemPtr, off) == 0) {
        count++;
      }
    }
  } else { // if slice attribute is specified
    int64_t numOfSlices = inputShape[sliceAxis];
    int64_t numOfElementsInSlice = 1;
    for (int64_t i = 0; i < inputRank; i++)
      numOfElementsInSlice *= (i == sliceAxis) ? 1 : inputShape[i];
    void *sliceDataPtr =
        (YPtr == NULL) ? alloca(numOfElementsInSlice * numOfSlices * dataSize)
                       : YPtr;
    if ((Y != NULL) && (indices == NULL)) {
      // temporal indices buffer is necessary to generate Y from indices
      indicesPtr = alloca(numOfSlices * sizeof(int64_t));
    }
    sliceTableInit(&sliceTable, dataType, sorted, numOfElementsInSlice,
        numOfSlices, sliceDataPtr, (uint64_t *)indicesPtr,
        (uint64_t *)inverseIndicesPtr, (uint64_t *)countsPtr);
    void *sliceData = alloca(numOfElementsInSlice * dataSize);
    for (int idxInSliceAxis = 0; idxInSliceAxis < numOfSlices;
         idxInSliceAxis++) {
      getSliceData(inputTensor, sliceAxis, idxInSliceAxis, sliceData);
      if (sliceTableRegister(&sliceTable, sliceData, idxInSliceAxis) == 0) {
        count++;
      }
    }
    if ((count > 0) && (Y != NULL)) {
      produceY(inputTensor, indices, sliceAxis, Y);
    }
  }
  *totalPtr = count;
#if _DEBUG_OMTENSOR
  printf("OMUnique: return (count = %d)\n", count);
  omTensorPrint("INPUT %t [\n%d", inputTensor);
  if (Y != NULL)
    omTensorPrint("] Y %t [\n%d", Y);
  if (indices != NULL)
    omTensorPrint("] INDICES %t [\n%d", indices);
  if (inverse_indices != NULL)
    omTensorPrint("] INV_INDICES %t [\n%d", inverse_indices);
  if (counts != NULL)
    omTensorPrint("] COUNTS %t [\n%d", counts);
  printf("]\n");
#endif
  return;
}

void omTensorUniqueCount(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t sliceAxis, uint64_t sorted) {
  omTensorUnique(
      totalTensor, NULL, NULL, NULL, NULL, inputTensor, sliceAxis, sorted);
  return;
}
