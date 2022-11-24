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

static inline int compareElement(void *dataPtr, uint64_t off1, uint64_t off2,
    uint64_t ascending, OM_DATA_TYPE dataType) {
  int result = 0;
  switch (dataType) {
  case ONNX_TYPE_BOOL:
    result = ascending ? (((bool *)dataPtr)[off1] > ((bool *)dataPtr)[off2])
                       : (((bool *)dataPtr)[off1] < ((bool *)dataPtr)[off2]);
    break;
  case ONNX_TYPE_UINT8:
    result = ascending
                 ? (((uint8_t *)dataPtr)[off1] > ((uint8_t *)dataPtr)[off2])
                 : (((uint8_t *)dataPtr)[off1] < ((uint8_t *)dataPtr)[off2]);
    break;
  case ONNX_TYPE_INT8:
    result = ascending
                 ? (((int8_t *)dataPtr)[off1] > ((int8_t *)dataPtr)[off2])
                 : (((int8_t *)dataPtr)[off1] < ((int8_t *)dataPtr)[off2]);
    break;
  case ONNX_TYPE_UINT16:
    result = ascending
                 ? (((uint16_t *)dataPtr)[off1] > ((uint16_t *)dataPtr)[off2])
                 : (((uint16_t *)dataPtr)[off1] < ((uint16_t *)dataPtr)[off2]);
    break;
  case ONNX_TYPE_INT16:
    result = ascending
                 ? (((int16_t *)dataPtr)[off1] > ((int16_t *)dataPtr)[off2])
                 : (((int16_t *)dataPtr)[off1] < ((int16_t *)dataPtr)[off2]);
    break;
  case ONNX_TYPE_UINT32:
    result = ascending
                 ? (((uint32_t *)dataPtr)[off1] > ((uint32_t *)dataPtr)[off2])
                 : (((uint32_t *)dataPtr)[off1] < ((uint32_t *)dataPtr)[off2]);
    break;
  case ONNX_TYPE_INT32:
    result = ascending
                 ? (((int32_t *)dataPtr)[off1] > ((int32_t *)dataPtr)[off2])
                 : (((int32_t *)dataPtr)[off1] < ((int32_t *)dataPtr)[off2]);
    break;
  case ONNX_TYPE_UINT64:
    result = ascending
                 ? (((uint64_t *)dataPtr)[off1] > ((uint64_t *)dataPtr)[off2])
                 : (((uint64_t *)dataPtr)[off1] < ((uint64_t *)dataPtr)[off2]);
    break;
  case ONNX_TYPE_INT64:
    result = ascending
                 ? (((int64_t *)dataPtr)[off1] > ((int64_t *)dataPtr)[off2])
                 : (((int64_t *)dataPtr)[off1] < ((int64_t *)dataPtr)[off2]);
    break;
  case ONNX_TYPE_FLOAT:
    result = ascending ? (((float *)dataPtr)[off1] > ((float *)dataPtr)[off2])
                       : (((float *)dataPtr)[off1] < ((float *)dataPtr)[off2]);
    break;
  case ONNX_TYPE_DOUBLE:
    result = ascending
                 ? (((double *)dataPtr)[off1] > ((double *)dataPtr)[off2])
                 : (((double *)dataPtr)[off1] < ((double *)dataPtr)[off2]);
    break;
  default:
    assert(false && "unexpected data type in compareElement");
  }
  return result;
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
  int64_t *order = (int64_t *)orderPtr;
  void *dataPtr = omTensorGetDataPtr(inputTensor);

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
            int64_t off = dim0 * strides[0] + dim1 * strides[1] +
                          dim2 * strides[2] + dim3 * strides[3] +
                          dim4 * strides[4];
            int64_t *idx = order + off;
            switch (algorithm) {
            case OMSORT_BUBBLE:
            default:
              // bubble sort
              for (uint32_t i = 0; i < sort_elems; i++) {
                for (uint32_t j = 0; j < sort_elems - 1; j++) {
                  if (compareElement(
                          dataPtr, idx[j], idx[j + 1], ascending, dataType)) {
                    int64_t tmp = idx[j];
                    idx[j] = idx[j + 1];
                    idx[j + 1] = tmp;
                  }
                }
              }
              break;
            case OMSORT_COMB: {
              // comb sort
              size_t gap = sort_elems;
              int swap = TRUE;
              while ((gap != 1) || swap) {
                gap = gap / 1.3;
                gap = (gap < 1) ? 1 : gap;
                swap = FALSE;
                for (uint32_t i = 0; i < sort_elems - gap; i++) {
                  if (compareElement(
                          dataPtr, idx[i], idx[i + gap], ascending, dataType)) {
                    int64_t tmp = idx[i];
                    idx[i] = idx[i + gap];
                    idx[i + gap] = tmp;
                    swap = TRUE;
                  }
                }
              }
              break;
            }
            }
          }
        }
      }
    }
  }
  return;
}
