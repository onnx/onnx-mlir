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
  uint64_t sorted;
  uint64_t elementSize;
  uint64_t elementMax;
  uint64_t elementCount;
  void *tablePtr;
  uint64_t *indicesPtr;
  uint64_t *inverseIndicesPtr;
  uint64_t *countsPtr;
} uniqueTable;

void uniqueTableInit(uniqueTable *table, OM_DATA_TYPE dataType, uint64_t sorted,
    uint64_t elemSize, uint64_t elemMax, void *tablePtr, uint64_t *indicesPtr,
    uint64_t *inverseIndicesPtr, uint64_t *countsPtr) {
  table->dataType = dataType;
  table->sorted = sorted;
  table->elementSize = elemSize;
  table->elementMax = elemMax;
  table->elementCount = 0;
  table->tablePtr = tablePtr;
  table->indicesPtr = indicesPtr;
  table->inverseIndicesPtr = inverseIndicesPtr;
  table->countsPtr = countsPtr;
}

void uniqueTablePrint(uniqueTable *table) {
  printf("uniqueTablePrint: dataType=%d, sorted=%ld, elemSize=%ld, elemMax=%ld, elemCount=%ld"
         "Table = [\n", table->dataType, table->sorted, table->elementSize, table->elementMax,
         table->elementCount);
  uint64_t dataSize = OM_DATA_TYPE_SIZE[table->dataType];  
  int numInSlice = table->elementSize / dataSize;
  for (uint64_t i = 0; i < table->elementCount; i++) {
    printf("  %ld:[", i);
    for (int j = 0; j < numInSlice; j++) {
      switch (table->dataType) {
      case ONNX_TYPE_INT64:
      case ONNX_TYPE_UINT64:
        printf("%ld,", ((int64_t *)(table->tablePtr))[i * numInSlice + j]);
        break;
      case ONNX_TYPE_FLOAT:
        printf("%f,", ((float *)(table->tablePtr))[i * numInSlice + j]);
        break;
      default:
        printf("XXX, ");
      }
    }
    printf("]\n");
  }
  printf("]\n");
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
    assert("Unsupported ONNX type in OMTensor");
  }
  return 0;
}

int isLessElem(
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

int uniqueTableRegister(uniqueTable *table, void *element, uint64_t off) {
  char *tablePtr = (char *)table->tablePtr;
  uint64_t insertIdx;
  int found = 0;
  // Searching for matching data in linear search
  // More optimizations, such as introducing hash table, are considerable.
  for (insertIdx = 0; insertIdx < table->elementCount; insertIdx++) {
    void *elementInTable = (void *)(tablePtr + table->elementSize * insertIdx);
    if (!memcmp(elementInTable, element, table->elementSize)) {
      found = 1;
      if (table->countsPtr != NULL)
        ((table->countsPtr)[insertIdx])++;
      break;
    }
    if (table->sorted && !isLessElem(elementInTable, element,
                             table->elementSize, table->dataType)) {
      break;
    }
  }
  if (found == 0) { // no matching element found in the table
    // make space to insert an element at insertIdx
    for (uint64_t j = table->elementCount; j > insertIdx; j--) {
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
        if ((table->inverseIndicesPtr)[j] >= insertIdx)
          (table->inverseIndicesPtr)[j] += 1;
      }
    }
    (table->elementCount)++;
    // insert the current element to the space
    void *elementInTable = (void *)(tablePtr + table->elementSize * insertIdx);
    memcpy(elementInTable, element, table->elementSize);
    if (table->countsPtr != NULL)
      (table->countsPtr)[insertIdx] = 1;
    if (table->indicesPtr != NULL)
      (table->indicesPtr)[insertIdx] = off;
  }
  if (table->inverseIndicesPtr != NULL)
    (table->inverseIndicesPtr)[off] = insertIdx;
  printf("uniqueTableRegister-");
  uniqueTablePrint(table);
  return found;
}

//
// Get slice data according to the specified slice axis(sliceAxis), and fill them to
// the specified pointer. Slice data's shape is the following.
// Input data: [x, y, z] (asumming sliceAxis= 1, the "y" axis is the slice.)
// Sliced data: [x, z, count] (s.t. count is numer of unique slice shape)
//
void getSliceData(void *inputPtr, const int64_t *inputShape, const int64_t *inputStrides,
    uint64_t inputRank, uint64_t sliceAxis, uint64_t idxInSliceAxis, uint64_t dataSize,
    void *sliceData) {
  assert(inputRank <= 6 && "rank should be 6 or less");
  assert(sliceAxis < inputRank && "rank should be less thatn rank");

  // To support input Tensor with various ranks in a uniform way.
  // If the input rank < 6, upgrade the rank to 6 virtually without changing
  // the physical memory layout by inserting length=1 ranks at lower ranks.
  uint64_t shape[6] = {1, 1, 1, 1, 1, 1};
  uint64_t shapeInUniqueAxis[6] = {1, 1, 1, 1, 1, 1};
  uint64_t strides[6] = {0, 0, 0, 0, 0, 0};
  int64_t sliceStride = 1;
  for (uint64_t i = 0; i < inputRank; i++) {
    shape[i] = inputShape[i];
    shapeInUniqueAxis[i] = (i == sliceAxis) ? 1 : inputShape[i];
    sliceStride *= shapeInUniqueAxis[i];
    strides[i] = inputStrides[i];
  }
#if 1
  printf("XXXX getSliceData: "
         "inputPtr=[%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld], "
         "shape=[%ld:%ld:%ld:%ld:%ld:%ld], "
         "strides=[%ld:%ld:%ld:%ld:%ld:%ld], sliceStride=%ld, rank=%ld, "
         "shapeInUniqueAxis=[%ld:%ld:%ld:%ld:%ld:%ld], "
         "shapeAxis=%ld, idxInSliceAxis=%ld, dataSize=%ld\n",
      ((uint64_t *)inputPtr)[0], ((uint64_t *)inputPtr)[1],
      ((uint64_t *)inputPtr)[2], ((uint64_t *)inputPtr)[3],
      ((uint64_t *)inputPtr)[4], ((uint64_t *)inputPtr)[5],
      ((uint64_t *)inputPtr)[6], ((uint64_t *)inputPtr)[7],
      ((uint64_t *)inputPtr)[8], ((uint64_t *)inputPtr)[9],
      ((uint64_t *)inputPtr)[10], ((uint64_t *)inputPtr)[11], shape[0],
      shape[1], shape[2], shape[3], shape[4], shape[5], strides[0], strides[1],
      strides[2], strides[3], strides[4], strides[5], sliceStride, inputRank,
      shapeInUniqueAxis[0], shapeInUniqueAxis[1], shapeInUniqueAxis[2],
      shapeInUniqueAxis[3], shapeInUniqueAxis[4], shapeInUniqueAxis[5], sliceAxis,
      idxInSliceAxis, dataSize);
#endif
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
              uint64_t off =
                  tdim[0] * strides[0] + tdim[1] * strides[1] +
                  tdim[2] * strides[2] + tdim[3] * strides[3] +
                  tdim[4] * strides[4] + tdim[5] * strides[5];
              void *currInputPtr = ((char *)inputPtr) + off * dataSize;
#if 1
              printf(
                  "getSliceData: idxInSliceAxis=%ld, tdim=[%ld:%ld:%ld:%ld:%ld:%ld], off=%ld, "
                  "strides=[%ld:%ld:%ld:%ld:%ld:%ld], sliceStride=%ld, (%ld) => outputIdx=%ld\n",
                  idxInSliceAxis, tdim[0], tdim[1], tdim[2], tdim[3], tdim[4], tdim[5], off,
                  strides[0], strides[1], strides[2], strides[3], strides[4],
                  strides[5], sliceStride, *((uint64_t *)currInputPtr), outputIdx);
#endif
              printf("AAAA %ld => %ld (%ld)\n", off, outputIdx, 
                     *((uint64_t *) currInputPtr));
              void *outputPtr = ((char *)sliceData) + outputIdx * dataSize;
              memcpy(outputPtr, currInputPtr, dataSize);
              outputIdx++;
            }
          }
        }
      }
    }
  }
#if 1
  printf("XXXX getSliceData returning : [");
  for (int i = 0; i < outputIdx; i++) {
    switch (dataSize) {
    case sizeof(uint64_t):
      printf("%ld,", ((uint64_t *)sliceData)[i]);
      break;
    case sizeof(float):
      printf("%f,", ((float *)sliceData)[i]);
      break;
    default:
      printf("XX, ");
      break;
    }
  }
  printf("] =============================================================\n");
#endif
}

#if 0
//
// Transpose the slice data into the original order. The shapes of sliced data and the input
// data are as follows. So that the last rank need to be back to the original position.
// Sliced data shape: [x, z, count] (s.t. count is numer of unique slice shape)
//           strides: [1, x, z]
// Input data shape: [x, count, z] (asumming sliceAxis= 1, the "y" axis is the slice.)
//          strides: [1, x, count]
//
void transposeY(OMTensor *Y, const int64_t *inputShape, const int64_t *inputStrides,
    uint64_t inputRank, uint64_t sliceAxis, int64_t count) {
  assert(inputRank <= 6 && "rank should be 6 or less");
  assert(sliceAxis < inputRank && "rank should be less thatn rank");
  const OM_DATA_TYPE dataType = omTensorGetDataType(Y);
  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];
  // To support input Tensor with various ranks in a uniform way.
  // If the input rank < 6, upgrade the rank to 6 virtually without changing
  // the physical memory layout by inserting length=1 ranks at lower ranks.
  // new axis becomes 
  uint64_t rank = 6;
  int64_t shape[6] = {1, 1, 1, 1, 1, 1};
  int64_t strides[6] = {1, 1, 1, 1, 1, 1};
  int64_t elementNum = 1;
  for (uint64_t i = 0; i < inputRank; i++) {
    shape[i] = (i == sliceAxis) ? 1 : inputShape[i];
    elementNum *= (i == sliceAxis) ? count : inputShape[i];
  }
  for (uint64_t i = 1; i < inputRank; i++)
    strides[i] = strides[i - 1] * shape[i - 1];
  int64_t sliceStrides = strides[sliceAxis];
  strides[sliceAxis] = count;

  printf("XXX transposeY elementNum=%ld\n", elementNum);
  // allocate a new buffer for input data, and copy them to the buffer.
  void *inputPtr = alloca(dataSize * elementNum);
  memcpy(inputPtr, omTensorGetDataPtr(Y), dataSize * elementNum);
  void *outputPtr = omTensorGetDataPtr(Y);
#if 1
  omTensorPrint("XXXX transposeY INPUT: Y= [\n", Y);
  switch (dataType) {
  case ONNX_TYPE_INT64:
  case ONNX_TYPE_UINT64:
    printf("]\nXXXX transposeY sliceAxis=%ld, "
           "inputPtr=[%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld], "
           "inputShape=[%ld:%ld:%ld:%ld:%ld:%ld], inputRank=%ld, inputAxis=%ld, "
           "shape=[%ld:%ld:%ld:%ld:%ld:%ld], rank=%ld, axis=%ld, "
           "strides=[%ld:%ld:%ld:%ld:%ld:%ld], sliceStrides=%ld, "
           "dataSize=%ld\n", sliceAxis,
           ((uint64_t *)inputPtr)[0], ((uint64_t *)inputPtr)[1],
           ((uint64_t *)inputPtr)[2], ((uint64_t *)inputPtr)[3],
           ((uint64_t *)inputPtr)[4], ((uint64_t *)inputPtr)[5],
           ((uint64_t *)inputPtr)[6], ((uint64_t *)inputPtr)[7],
           ((uint64_t *)inputPtr)[8], ((uint64_t *)inputPtr)[9],
           ((uint64_t *)inputPtr)[10], ((uint64_t *)inputPtr)[11],
           inputShape[0], inputShape[1], inputShape[2], inputShape[3], inputShape[4], inputShape[5],
           inputRank, sliceAxis,
           shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
           rank, sliceAxis,
           strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
           sliceStrides, dataSize);
    break;
  case ONNX_TYPE_FLOAT:
    printf("]\nXXXX transposeY sliceAxis=%ld, "
           "inputPtr=[%f:%f:%f:%f:%f:%f:%f:%f:%f:%f:%f:%f], "
           "inputShape=[%ld:%ld:%ld:%ld:%ld:%ld], inputRank=%ld, inputAxis=%ld, "
           "shape=[%ld:%ld:%ld:%ld:%ld:%ld], rank=%ld, axis=%ld, "
           "strides=[%ld:%ld:%ld:%ld:%ld:%ld], sliceStrides=%ld, "
           "dataSize=%ld\n", sliceAxis,
           ((float *)inputPtr)[0], ((float *)inputPtr)[1],
           ((float *)inputPtr)[2], ((float *)inputPtr)[3],
           ((float *)inputPtr)[4], ((float *)inputPtr)[5],
           ((float *)inputPtr)[6], ((float *)inputPtr)[7],
           ((float *)inputPtr)[8], ((float *)inputPtr)[9],
           ((float *)inputPtr)[10], ((float *)inputPtr)[11],
           inputShape[0], inputShape[1], inputShape[2], inputShape[3], inputShape[4], inputShape[5],
           inputRank, sliceAxis,
           shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
           rank, sliceAxis,
           strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
           sliceStrides, dataSize);
    break;
  default:
    break;
  }
  fflush(stdout);
#endif

  int64_t dim[6];
  int64_t inputIdx = 0;
  for (int64_t idxInSliceAxis = 0; idxInSliceAxis < count; idxInSliceAxis++) {
    for (dim[5] = 0; dim[5] < shape[5]; dim[5]++) {
      for (dim[4] = 0; dim[4] < shape[4]; dim[4]++) {
        for (dim[3] = 0; dim[3] < shape[3]; dim[3]++) {
          for (dim[2] = 0; dim[2] < shape[2]; dim[2]++) {
            for (dim[1] = 0; dim[1] < shape[1]; dim[1]++) {
              for (dim[0] = 0; dim[0] < shape[0]; dim[0]++) {
                // calculate the input and output pointers
                void *InputPtr = ((char *)inputPtr) + inputIdx * dataSize;
                uint64_t off = idxInSliceAxis * sliceStrides +
                    dim[0] * strides[0] + dim[1] * strides[1] +
                    dim[2] * strides[2] + dim[3] * strides[3] +
                    dim[4] * strides[4] + dim[5] * strides[5];
                void *OutputPtr = ((char *)outputPtr) + off * dataSize;
#if 1
                printf("YYYY axis=%ld, idxInSliceAxis=%ld, dim=[%ld:%ld:%ld:%ld:%ld:%ld], "
                    "strides=[%ld:%ld:%ld:%ld:%ld:%ld]: off=%ld\n", sliceAxis,
                    idxInSliceAxis, dim[0], dim[1], dim[2], dim[3], dim[4], dim[5],
                    strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
                    off);
#endif
                if (dataSize == 8) {
                  printf("ZZZZ %ld=>%ld (%ld)\n", inputIdx, off, ((uint64_t *)InputPtr)[0]);
                } else {
                  printf("ZZZZ %ld=>%ld (%f)\n", inputIdx, off, ((float *)InputPtr)[0]);
                }
                fflush(stdout);
                memcpy(OutputPtr, InputPtr, dataSize);
                inputIdx++;
              }
            }
          }
        }
      }
    }
  }
#if 1
  omTensorPrint("XXXX transposeY OUTPUT: Y= [\n", Y);
  printf("]\nXXXX transposeY sliceAxis=%ld, inputPtr=[%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld:%ld], "
         "shape=[%ld:%ld:%ld:%ld:%ld:%ld], "
         "strides=[%ld:%ld:%ld:%ld:%ld:%ld], rank=%ld, "
         "axis=%ld, dataSize=%ld\n", sliceAxis,
      ((uint64_t *)inputPtr)[0], ((uint64_t *)inputPtr)[1],
      ((uint64_t *)inputPtr)[2], ((uint64_t *)inputPtr)[3],
      ((uint64_t *)inputPtr)[4], ((uint64_t *)inputPtr)[5],
      ((uint64_t *)inputPtr)[6], ((uint64_t *)inputPtr)[7],
      ((uint64_t *)inputPtr)[8], ((uint64_t *)inputPtr)[9],
      ((uint64_t *)inputPtr)[10], ((uint64_t *)inputPtr)[11], shape[0],
      shape[1], shape[2], shape[3], shape[4], shape[5], strides[0], strides[1],
      strides[2], strides[3], strides[4], strides[5], rank, sliceAxis, dataSize);
  fflush(stdout);
#endif
}
#endif

//
// Produce OMTensor Y from input tensor, indices and sliceAxis
//
void produceY(const OMTensor *inputTensor, OMTensor *indices, int64_t sliceAxis, OMTensor *Y) {
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
  int64_t *indicesPtr = (int64_t *) omTensorGetDataPtr(indices);
  assert(count > 0 && "count should be greater than 0");
  void *YPtr = omTensorGetDataPtr(Y);
  
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];

#if 1
  printf("XXXX produceY:\n");
  omTensorPrint("inputTensor=[\n", inputTensor);
  omTensorPrint("], indices=[\n", indices);
  printf("], sliceAxis=%ld\n", sliceAxis);
  fflush(stdout);
#endif

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
    strides[i] = inputStrides[i]; //(i == 0) ? 1 : (strides[i - 1] * inputShape[i - 1]);
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
              int64_t tdim[6] = {dim[0], dim[1], dim[2], dim[3], dim[4], dim[5]};
              tdim[sliceAxis] = indicesPtr[dim[sliceAxis]];
              // calculate the input offset and pointer
              uint64_t inputOff =
                tdim[0] * strides[0] + tdim[1] * strides[1] +
                tdim[2] * strides[2] + tdim[3] * strides[3] +
                tdim[4] * strides[4] + tdim[5] * strides[5];
              void *inputNumPtr = ((char *)inputPtr) + inputOff * dataSize;
              // calculate the output offset and pointer
              uint64_t outputOff = 
                dim[0] * outStrides[0] + dim[1] * outStrides[1] +
                dim[2] * outStrides[2] + dim[3] * outStrides[3] +
                dim[4] * outStrides[4] + dim[5] * outStrides[5];
              void *YNumPtr = ((char *)YPtr) + outputOff * dataSize;
#if 1
              printf("YYY  input: dim=[%ld:%ld:%ld:%ld:%ld:%ld] strides=[%ld:%ld:%ld:%ld:%ld:%ld], dim[sliceAxis]=%ld => indicesPtr[dim[sliceAxis]]=%ld\n",
                     tdim[0], tdim[1], tdim[2], tdim[3], tdim[4], tdim[5],
                     strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
                     dim[sliceAxis], indicesPtr[dim[sliceAxis]]);
              printf("YYY output: dim=[%ld:%ld:%ld:%ld:%ld:%ld] strides=[%ld:%ld:%ld:%ld:%ld:%ld]\n",
                     dim[0], dim[1], dim[2], dim[3], dim[4], dim[5],
                     outStrides[0], outStrides[1], outStrides[2], outStrides[3], outStrides[4],
                     outStrides[5]);
#endif
              if (dataSize == 8) {
                printf("ZZZ %ld=>%ld (%ld)\n", inputOff, outputOff, ((uint64_t *)inputNumPtr)[0]);
              } else {
                printf("ZZZ %ld=>%ld (%f)\n", inputOff, outputOff, ((float *)inputNumPtr)[0]);
              }
              fflush(stdout);
              memcpy(YNumPtr, inputNumPtr, dataSize);
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
  assert(inputRank <= 6 && "input rank should be 6 or less");
  assert(sliceAxis < inputRank && "axis should be less than rank");
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
    uniqueTableInit(&uniqueTable, dataType, sorted, dataSize, elementNum,
        tablePtr, (uint64_t *)indicesPtr, (uint64_t *)inverseIndicesPtr,
        (uint64_t *)countsPtr);
    for (uint64_t off = 0; off < elementNum; off++) {
      void *elemPtr = ((char *)inputPtr) + dataSize * off;
      if (uniqueTableRegister(&uniqueTable, elemPtr, off) == 0) {
        count++;
      }
    }
  } else { // if slice attribute is specified
    int64_t numOfSlices = inputShape[sliceAxis];
    int64_t numOfElementsInSlice = 1;
    for (int64_t i = 0; i < inputRank; i++)
      numOfElementsInSlice *= (i == sliceAxis) ? 1 : inputShape[i];
    void *tablePtr =
        (YPtr == NULL) ? alloca(numOfElementsInSlice * numOfSlices * dataSize) : YPtr;
    uniqueTableInit(&uniqueTable, dataType, sorted, numOfElementsInSlice * dataSize,
        numOfSlices, tablePtr, (uint64_t *)indicesPtr,
        (uint64_t *)inverseIndicesPtr, (uint64_t *)countsPtr);
    void *sliceData = alloca(numOfElementsInSlice * dataSize);
    for (int idxInSliceAxis = 0; idxInSliceAxis < numOfSlices; idxInSliceAxis++) {
      getSliceData(inputPtr, inputShape, inputStrides, inputRank, sliceAxis, idxInSliceAxis,
          dataSize, sliceData);
      if (uniqueTableRegister(&uniqueTable, sliceData, idxInSliceAxis) == 0) {
        count++;
      }
    }
    if (Y != NULL) {
      produceY(inputTensor, indices, sliceAxis, Y);
    }
  }
  *totalPtr = count;
  printf("XXXX: count=%d\n", count);
#if 1
  printf("OMUnique: return (count = %d)\n", count);
  omTensorPrint("INPUT [\n", inputTensor);
  if (Y != NULL)
    omTensorPrint("] Y [\n", Y);
  if (indices != NULL)
    omTensorPrint("] INDICES [\n", indices);
  if (inverse_indices != NULL)
    omTensorPrint("] INV_INDICES [\n", inverse_indices);
  if (counts != NULL)
    omTensorPrint("] COUNTS [\n", counts);
  printf("]\n");
#endif
  return;
}

void omTensorUniqueCount(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t sliceAxis, uint64_t sorted) {
  omTensorUnique(
      totalTensor, inputTensor, sliceAxis, sorted, NULL, NULL, NULL, NULL);
  return;
}
