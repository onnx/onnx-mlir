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
typedef struct elementTable {
  void *tablePointer;
  uint64_t elementSize;
  uint64_t elementMax;
  uint64_t elementCount;
} elementTable;

#define elementTableInit(table, elemSize, elemMax) \
do { \
  (table)->tablePointer = alloca((elemSize) * (elemMax)); \
  (table)->elementSize = (elemSize); \
  (table)->elementMax = (elemMax); \
  (table)->elementCount = 0; \
} while (0)

int elementTableRegister(elementTable *table, void *element) {
  char *tablePtr = (char *)table->tablePointer;
  uint64_t i;
  for (i = 0; i < table->elementCount; i++) {
    void *elementInTable = (void *)(tablePtr + table->elementSize * i);
    if (!memcmp(elementInTable, element, table->elementSize)) {
      break;
    }
  }
  if (i == table->elementCount) { // no matching element found in the table
    void *elementInTable = (void *)(tablePtr + table->elementSize * i);
    memcpy(elementInTable, element, table->elementSize);
    (table->elementCount)++;
    return 0;
  }
  return 1;
}

uint64_t elementTable_count(elementTable *table) { return table->elementCount; }

void omTensorUnique(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t inputAxis, uint64_t sorted, OMTensor *Y, OMTensor *indices,
    OMTensor *reverse_indices, OMTensor *counts) {
  printf("omTensorUnique: axis=%ld, sorted=%ld, Y=%p, indices=%p, "
         "reverse_indices=%p, count=%p, ", inputAxis, sorted, Y, indices,
         reverse_indices, counts);
  omTensorPrint("input:", inputTensor);
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  const int64_t inputRank = omTensorGetRank(inputTensor);
  assert(inputRank <= 6 && "omTensorUnique assumes inputRank <= 6");
  int64_t *totalPtr = (int64_t *)omTensorGetDataPtr(totalTensor);
  const int64_t *inputShape = omTensorGetShape(inputTensor);
  const int64_t *inputStrides = omTensorGetStrides(inputTensor);
  void *dataPtr = omTensorGetDataPtr(inputTensor);
  void *YPtr = (Y != NULL) ? omTensorGetDataPtr(Y) : NULL;
  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];

  int count = 0;
  if (inputAxis < 0) { // manage the inputTensor as flatten one
    uint64_t elementNum = 1;
    for (int64_t i = 0; i < inputRank; i++) {
      elementNum *= inputShape[i];
    }
    elementTable elementTable;
    elementTableInit(&elementTable, dataSize, elementNum);
    for (uint64_t off = 0; off < elementNum; off++) {
      void *data = ((char *)dataPtr) + dataSize * off;
      if (elementTableRegister(&elementTable, data) == 0) {
        if (YPtr != NULL) {
          void *currY = ((char *)YPtr) + dataSize * count;
          memcpy(currY, data, dataSize);
        }
        count++;
      }
    }
  } else {
    // To support input Tensor with various ranks in a uniform way.
    // If the input rank < 6, upgrade the rank to 6 virtually without changing
    // the physical memory layout by inserting length=1 ranks at lower ranks.
    // The 5th axis becomes the unique axis.
    int64_t shape[6] = {1, 1, 1, 1, 1, 1};
    int64_t shapeInUniqueAxis[6] = {1, 1, 1, 1, 1, 1};
    int64_t strides[6] = {0, 0, 0, 0, 0, 0};
    int64_t rank = 6;
    int64_t axis = inputAxis + (rank - inputRank);
    for (int64_t i = 0; i < inputRank; i++) {
      shape[i + (rank - inputRank)] = inputShape[i];
      shapeInUniqueAxis[i + (rank - inputRank)] =
          (i == axis) ? 1 : inputShape[i];
      strides[i + (rank - inputRank)] = inputStrides[i];
    }
    int dimCurrInUniqueAxis, dimPrevInUniqueAxis;
    for (dimCurrInUniqueAxis = 0; dimCurrInUniqueAxis < shape[axis];
         dimCurrInUniqueAxis++) {
      for (dimPrevInUniqueAxis = 0; dimPrevInUniqueAxis < dimCurrInUniqueAxis;
           dimPrevInUniqueAxis++) {
        // compare prev and curr values to check if the curr values are unique
        // or not.
        int dim[6];
        for (dim[0] = 0; dim[0] < shapeInUniqueAxis[0]; dim[0]++) {
          for (dim[1] = 0; dim[1] < shapeInUniqueAxis[1]; dim[1]++) {
            for (dim[2] = 0; dim[2] < shapeInUniqueAxis[2]; dim[2]++) {
              for (dim[3] = 0; dim[3] < shapeInUniqueAxis[3]; dim[3]++) {
                for (dim[4] = 0; dim[4] < shapeInUniqueAxis[4]; dim[4]++) {
                  for (dim[5] = 0; dim[5] < shapeInUniqueAxis[5]; dim[5]++) {
                    int dimCurr[6], dimPrev[6];
                    for (int i = 0; i < 6; i++) {
                      dimCurr[i] = dim[i];
                      dimPrev[i] = dim[i];
                    }
                    dimCurr[axis] = dimCurrInUniqueAxis;
                    dimPrev[axis] = dimPrevInUniqueAxis;
                    uint64_t offCurr =
                        dimCurr[0] * strides[0] + dimCurr[1] * strides[1] +
                        dimCurr[2] * strides[2] + dimCurr[3] * strides[3] +
                        dimCurr[4] * strides[4] + dimCurr[5] * strides[5];
                    uint64_t offPrev =
                        dimPrev[0] * strides[0] + dimPrev[1] * strides[1] +
                        dimPrev[2] * strides[2] + dimPrev[3] * strides[3] +
                        dimPrev[4] * strides[4] + dimPrev[5] * strides[5];
                    void *dataCurr = ((char *)dataPtr) + dataSize * offCurr;
                    void *dataPrev = ((char *)dataPtr) + dataSize * offPrev;
                    if (memcmp(dataCurr, dataPrev, dataSize)) {
                      goto mismatch_element_found;
                    }
                  }
                }
              }
            }
          }
        }
      mismatch_element_found:
        if (dim[0] == shapeInUniqueAxis[0]) {
          // the previous and curr values in the specified axis are the same
          break;
        }
      }
      if (dimPrevInUniqueAxis == dimCurrInUniqueAxis) {
        //if (YPtr != NULL) {
          //void *currY = ((char *)YPtr) + dataSize * count;
          //memcpy(currY, data, dataSize);
        //}
        // TODO: Fill values of return variables, such as Y, indices,
        //       inverse_indices and counts
        count++; // dimCurrInUniqueAxis is a unique dim in the unique axis
      }
    }
  }
  *totalPtr = count;
  printf("==> %d\n", count);
  //omTensorPrint("omTensorUnique: total =", totalTensor);
  if (YPtr != NULL) {
    omTensorPrint("omTensorUnique: Y =", Y);
  }
  printf("======================================================================\n");
  return;
}

void omTensorUniqueCount(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t inputAxis, uint64_t sorted) {
  omTensorUnique(totalTensor, inputTensor, inputAxis, sorted, NULL, NULL, NULL, NULL);
  return;
}
