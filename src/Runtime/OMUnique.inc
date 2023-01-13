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
#include "src/Runtime/OMUnique.h"
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

void elementTableInit(
    elementTable *table, uint64_t elementSize, uint64_t elementMax) {
  table->tablePointer = alloca(elementSize * elementMax);
  table->elementSize = elementSize;
  table->elementMax = elementMax;
  table->elementCount = 0;
}

int elementTableRegister(elementTable *table, void *element) {
  char *tablePtr = (char *)table->tablePointer;
  int i;
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

#if 0
//
// Table for matrix
//
typedef struct matrixTable {
  void *tablePointer;
  uint64_t uniqueShape[OMUNIQUE_MAXRANK];
  uint64_t uniqueShapeBegin[OMUNIQUE_MAXRANK];
  uint64_t uniqueShapeEnd[OMUNIQUE_MAXRANK];
  uint64_t matrixSize;
  int uniqueAxis;
  uint64_t matrixMax;
  uint64_t matrixCount;
} matrixTable;

void matrixTableInit(matrixTable *table, int matrixRank, uint64_t *matrixShape,
     uint64_t dataSize, uint64_t uniqueAxis) {
  uint64_t matrixSize = dataSize;
  for (int i = 0; i < OMUNIQUE_MAXRANK; i++) {
    table->uniqueShape[i] = matrixShape[i];
    table->uniqueShapeBegin[i] = 0;
    if (i == uniqueAxis) {
      table->uniqueShapeEnd[i] = 0;
    } else {
      table->uniqueShapeEnd[i] = matrixShape[i];
      matrixSize *= matrixShape[i];
    }
  }
  table->tablePointer = alloca(dataSize * matrixMax);
  table->uniqueShape = matrixShape;
  table->uniqueShape = matrixShape;
  table->uniqueAxis = uniqueAxis;
  table->dataSize = dataSize;
  table->matrixMax = matrixShape[axis];
  table->matrixCount = 0;
}

int matrixTableRegister(matrixTable *table, void *dataPtr, int64_t idx) {
  char *tablePtr = table->matrixpointer;
  int uniqueRank = table->uniqueRank;
  uint64_t *shape = table->matrixShape;
  uint64_t *shapeBegin = table->matrixShapeBegin;
  uint64_t *shapeEnd = table->matrixShapeEnd;
  for (uint64_t i = 0; i < table->matrixCount; i++) {
    void *matrixInTable = (void *) (tablePtr + table->matrixSize * i);
    for (int j = 0; j < table->uniqueRank; j++) {
      for (int dim0 = shapeBegin[0] < shapeEnd[0]; dim0++) {
        for (int dim1 = shapeBegin[1]; dim1 < shapeEnd[1]; dim1++) {
          for (int dim2 = shapeBegin[2]; dim2 < shapeEnd[2]; dim2++) {
            for (int dim3 = shapeBegin[3]; dim3 < shapeEnd[3]; dim3++) {
              for (int dim4 = shapeBegin[4]; dim4 < shapeEnd[4]; dim4++) {
                for (int dim5 = shapeBegin[5]; dim5 < shapeEnd[5]; dim5++) {
                uint64_t off = dim0 * strides[0] + dim1 * strides[1] +
                               dim2 * strides[2] + dim3 * strides[3] +
                               dim4 * strides[4] + dim5 * strides[5];
                void *data = ((char *)dataPtr) + table->dataSize * off;
                if (elementTableRegister(&elementTable, data) == 0)
                  count++; 
              }
            }
          }
        }
      }
    }
    for (uint64_t rank = 0; rank < table->matrixRank; rank++) {
       if (rank != table->matrixRank) {
         int64_t off = dim0 * strides[0] + dim1 * strides[1] +
                       dim2 * strides[2] + dim3 * strides[3] +
                       dim4 * strides[4] + dim5 * strides[5];
         void *data = ((char *)dataPtr) + matrixSize * off;
{
         && !memcmp(matrix 
       }
    }
    if (j == table->matrixRank) {
      // matching matrix found in the table
      break;
    }
  }
  if (i == table->matrixCount) { // no matching matrix found in the table
    void *elemInTable = (void *) (tablePtr + table->matrixSize * i);
    memcpy(elemInTable, element, table->matrixSize);
    (table->elementCount)++;
    return 0;
  }
  return 1;
}
#endif

int omTensorUnique(
    const OMTensor *inputTensor, uint64_t axis, uint64_t sorted, int flag) {
  omTensorPrint("omTensorUnique: ", inputTensor);
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  const uint64_t rank = omTensorGetRank(inputTensor);
  assert(rank <= 6 && "omTensorUnique assumes rank <= 6");
  const int64_t *inputShape = omTensorGetShape(inputTensor);
  const int64_t *inputStrides = omTensorGetStrides(inputTensor);
  void *dataPtr = omTensorGetDataPtr(inputTensor);
  int unique_elems = inputShape[axis];
  // Uniqueing not necessary for array whose size <= 1
  if (unique_elems <= 1)
    return unique_elems;

  uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];

  // To support input Tensor with various ranks in a uniform way.
  // If the input rank < 6, upgrade the rank to 6 virtually without changing
  // the physical memory layout by inserting length=1 ranks at lower ranks.
  // The 5th axis becomes the unique axis.
  int64_t shape[6] = {1, 1, 1, 1, 1, 1};
  int64_t shapeInUniqueAxis[6] = {1, 1, 1, 1, 1, 1};
  int64_t strides[6] = {0, 0, 0, 0, 0, 0};
  uint64_t elementnum = 1;
  for (uint64_t i = 0; i < rank; i++) {
    shape[i + (6 - rank)] = inputShape[i];
    shapeInUniqueAxis[i + (6 - rank)] = (i == axis) ? 1 : inputShape[i];
    strides[i + (6 - rank)] = inputStrides[i];
    elementnum *= inputShape[i];
  }

  int count = 0;
  if (axis < 0) { // manage the inputTensor as flatten one
    elementTable elementTable;
    elementTableInit(&elementTable, elementnum, dataSize);
    for (int dim0 = 0; dim0 < shape[0]; dim0++) {
      for (int dim1 = 0; dim1 < shape[1]; dim1++) {
        for (int dim2 = 0; dim2 < shape[2]; dim2++) {
          for (int dim3 = 0; dim3 < shape[3]; dim3++) {
            for (int dim4 = 0; dim4 < shape[4]; dim4++) {
              for (int dim5 = 0; dim5 < shape[5]; dim5++) {
                uint64_t off = dim0 * strides[0] + dim1 * strides[1] +
                               dim2 * strides[2] + dim3 * strides[3] +
                               dim4 * strides[4] + dim5 * strides[5];
                void *data = ((char *)dataPtr) + dataSize * off;
                if (elementTableRegister(&elementTable, data) == 0) {
                  count++;
                }
              }
            }
          }
        }
      }
    }
  } else {
    int dimCurrInUniqueAxis, dimPrevInUniqueAxis;
    for (dimCurrInUniqueAxis = 0; dimCurrInUniqueAxis < shape[axis];
         dimCurrInUniqueAxis++) {
      for (dimPrevInUniqueAxis = 0; dimPrevInUniqueAxis < dimCurrInUniqueAxis;
           dimPrevInUniqueAxis++) {
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
                      goto mismatch_found;
                    }
                  }
                }
              }
            }
          }
        }
      mismatch_found:
        continue;
      }
      if (dimPrevInUniqueAxis == dimCurrInUniqueAxis) {
        count++; // new matrix found
      }
    }
  }
  return count;
}
