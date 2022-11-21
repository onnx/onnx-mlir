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
  assert(rank == 3 && "omTensorSort assumes rank == 3");
  assert(axis == (rank - 1) && "omTensorSort assumes axis == (rank - 1)");
  const int64_t *shape = omTensorGetShape(inputTensor);
  const int64_t *strides = omTensorGetStrides(inputTensor);
  void *orderPtr = omTensorGetDataPtr(orderTensor);
  void *dataPtr = omTensorGetDataPtr(inputTensor);

  int64_t *order = (int64_t *)orderPtr;
  float *data = (float *)dataPtr;
  int64_t total_elems = shape[axis];

  for (int dim0 = 0; dim0 < shape[0]; dim0++) {
    for (int dim1 = 0; dim1 < shape[1]; dim1++) {
      int64_t *idx = order + dim0 * shape[1] * shape[2] + dim1 * shape[2];
      switch (algorithm) {
      case OMSORT_BUBBLE:
      default:
        // bubble sort
        for (uint32_t i = 0; i < total_elems; i++) {
          for (uint32_t j = 0; j < total_elems - 1; j++) {
            if (compareElement(data, idx[j], idx[j + 1], ascending, dataType)) {
              int64_t tmp = idx[j];
              idx[j] = idx[j + 1];
              idx[j + 1] = tmp;
            }
          }
        }
        break;
      case OMSORT_COMB: {
        // comb sort
        size_t gap = total_elems;
        int swap = TRUE;
        while ((gap != 1) || swap) {
          gap = gap / 1.3;
          gap = (gap < 1) ? 1 : gap;
          swap = FALSE;
          for (uint32_t i = 0; i < total_elems - gap; i++) {
            if (compareElement(
                    data, idx[i], idx[i + gap], ascending, dataType)) {
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
  return;
}
