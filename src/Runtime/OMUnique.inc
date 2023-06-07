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
  uint64_t elementSize;
  uint64_t elementMax;
  uint64_t elementCount;
  void *tablePtr;
  int64_t *indicesPtr;
  int64_t *inverse_indicesPtr;
  int64_t *countsPtr;
} elementTable;

#define elementTableInit(table, elemSize, elemMax, tablePtr, indicesPtr, \
                         inverse_indicesPtr, countsPtr) \
do { \
  (table)->elementSize = (elemSize); \
  (table)->elementMax = (elemMax); \
  (table)->elementCount = 0; \
  (table)->tablePtr = (tablePtr); \
  (table)->indicesPtr = (int64_t *) (indicesPtr); \
  (table)->inverse_indicesPtr = (int64_t *) (inverse_indicesPtr); \
  (table)->countsPtr = (int64_t *) (countsPtr); \
} while (0)

int elementTableRegister(elementTable *table, void *element, uint64_t off) {
  char *tablePtr = (char *)table->tablePtr;
  uint64_t i;
  int found = 0;
  for (i = 0; i < table->elementCount; i++) {
    void *elementInTable = (void *)(tablePtr + table->elementSize * i);
    if (!memcmp(elementInTable, element, table->elementSize)) {
      found = 1;
      break;
    }
  }
  if (i == table->elementCount) { // no matching element found in the table
    void *elementInTable = (void *)(tablePtr + table->elementSize * i);
    memcpy(elementInTable, element, table->elementSize);
    if (table->countsPtr != NULL)
      (table->countsPtr)[table->elementCount] = 0;
    (table->elementCount)++;
  }
  if (table->indicesPtr != NULL)
    (table->indicesPtr)[i] = off;
  if (table->inverse_indicesPtr != NULL)
    (table->inverse_indicesPtr)[off] = i;
  if (table->countsPtr != NULL)
    ((table->countsPtr)[table->elementCount])++;
  return found;
}

int64_t elementTable_count(elementTable *table) { return table->elementCount; }

void omTensorUnique(OMTensor *totalTensor, const OMTensor *inputTensor,
   int64_t inputAxis, uint64_t sorted, OMTensor *Y, OMTensor *indices,
   OMTensor *inverse_indices, OMTensor *counts) {
 //printf("YYYY: omTensorUnique called(axis=%ld, sorted=%ld, Y=%p, indices=%p, "
 //       "inverse_indices=%p, count=%p)\n", inputAxis, sorted, Y, indices,
 //       inverse_indices, counts);
 //omTensorPrint("  input:", inputTensor); fflush(stdout);
 const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
 const int64_t inputRank = omTensorGetRank(inputTensor);
 assert(inputRank <= 6 && "omTensorUnique assumes inputRank <= 6");
 int64_t *totalPtr = (int64_t *)omTensorGetDataPtr(totalTensor);
 const int64_t *inputShape = omTensorGetShape(inputTensor);
 const int64_t *inputStrides = omTensorGetStrides(inputTensor);
 void *inputPtr = omTensorGetDataPtr(inputTensor);
 void *YPtr = (Y != NULL) ? omTensorGetDataPtr(Y) : NULL;
 void *indicesPtr = (indices != NULL) ? omTensorGetDataPtr(indices) : NULL;
 void *inverse_indicesPtr = (inverse_indices != NULL) ? omTensorGetDataPtr(inverse_indices) : NULL;
 void *countsPtr = (counts != NULL) ? omTensorGetDataPtr(counts) : NULL;
 uint64_t dataSize = OM_DATA_TYPE_SIZE[dataType];

 int count = 0;
 if (inputAxis < 0) { // manage the inputTensor as flatten one
   uint64_t elementNum = 1;
   for (int64_t i = 0; i < inputRank; i++) {
     elementNum *= inputShape[i];
   }
   elementTable elementTable;
   void *tablePtr = (YPtr == NULL) ? alloca(dataSize * elementNum) : YPtr;
   elementTableInit(&elementTable, dataSize, elementNum, tablePtr, indicesPtr,
                    inverse_indicesPtr, countsPtr);
   for (uint64_t off = 0; off < elementNum; off++) {
     void *elemPtr = ((char *)inputPtr) + dataSize * off;
     if (elementTableRegister(&elementTable, elemPtr, off) == 0) {
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
   int ansSize = dataSize * strides[5];
   void *currAns = alloca(ansSize);
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
                   void *dataCurr = ((char *)inputPtr) + dataSize * offCurr;
                   void *dataPrev = ((char *)inputPtr) + dataSize * offPrev;
                   if (memcmp(dataCurr, dataPrev, dataSize)) {
                     goto mismatch_element_found;
                   }
                   void *currAnsData = ((char *)currAns) +
                       dimCurrInUniqueAxis * dataSize;
                   memcpy(currAnsData, dataCurr, dataSize);
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
       if (YPtr != NULL) {
         void *currY = ((char *)YPtr) + dataSize * count;
           memcpy(currY, currAns, ansSize);
         }
       }
       // TODO: Fill values of return variables, such as Y, indices,
       //       inverse_indices and counts
       count++; // dimCurrInUniqueAxis is a unique dim in the unique axis
    }
  }
  *totalPtr = count;
  //printf("==> %d\n", count);
  //omTensorPrint("omTensorUnique: total =", totalTensor);
  //if (YPtr != NULL) {
  //  omTensorPrint("omTensorUnique: Y =", Y);
  //}
  //printf("======================================================================\n");
  return;
}

void omTensorUniqueCount(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t inputAxis, uint64_t sorted) {
  omTensorUnique(totalTensor, inputTensor, inputAxis, sorted, NULL, NULL, NULL, NULL);
  return;
}
