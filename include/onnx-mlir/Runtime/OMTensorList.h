//===-------------- OMTensorList.h - OMTensorList Declaration header
//--------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMTensorList data structures and
// API functions.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OMTENSORLIST_H
#define ONNX_MLIR_OMTENSORLIST_H

#include "onnx-mlir/Runtime/OMTensor.h"

struct OMTensorList;

#ifndef __cplusplus
typedef struct OMTensorList OMTensorList;
#endif

/**
 * \brief OMTensorList creator
 *
 * Create an OMTensorList with specified OMTensor array. The array of pointers
 * to OMTensor pointers is used without copying, so caller should not free the
 * `tensors` ptr.
 *
 * @param tensors array of pointers to OMTensor
 * @param n number of elements in tensors array
 * @return pointer to the OMTensorList created, NULL if creation failed.
 *
 */
OMTensorList *omTensorListCreate(OMTensor **tensors, int n);

/**
 * \brief OMTensorList destroyer
 *
 * Destroy the OMTensorList struct recursively. That is to say, both the
 * ptr to the OMTensor pointers AND the OMTensor pointers are freed.
 *
 * @param list pointer to the OMTensorList to be destroyed
 *
 */
void omTensorListDestroy(OMTensorList *list);

/**
 * \brief OMTensorList OMTensor array getter
 *
 * The pointer to OMTensor pointers are returned without copying, so caller
 * should not free the returned pointer.
 *
 * @param list pointer to the OMTensorList
 * @return pointer to the array of OMTensor pointers.
 */
OMTensor **omTensorListGetPtrToOmts(OMTensorList *list);

/**
 * \brief OMTensorList size getter
 *
 *
 * @param list pointer to the OMTensorList
 * @return number of elements in the OMTensor array.
 */
int omTensorListGetSize(OMTensorList *list);

/**
 * \brief OMTensorList OMTensor getter by index
 *
 * @param list pointer to the OMTensorList
 * @param index index of the OMTensor
 * @reutrn pointer to the OMTensor, NULL if not found.
 */
OMTensor *omTensorListGetOmtByIndex(OMTensorList *list, size_t index);

#endif // ONNX_MLIR_OMTENSORLIST_H
