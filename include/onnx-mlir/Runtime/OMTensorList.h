/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- OMTensorList.h - OMTensorList Declaration header-------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief OMTensorList creator
 *
 * Create an OMTensorList with specified OMTensor array.
 *
 * Internally OMTensorList creates its own copy of the passed-in tensor pointer
 * array. This copy is freed when the OMTensorList is destroyed.
 *
 * @param tensors array of pointers to OMTensor
 * @param n number of elements in tensors array
 * @return pointer to the OMTensorList created, NULL if creation failed.
 *
 */
OM_EXTERNAL_VISIBILITY OMTensorList *omTensorListCreate(
    OMTensor **tensors, int64_t n);

/**
 * \brief OMTensorList creator with shallow
 *
 * Create an OMTensorList with specified OMTensor array.
 *
 * Internally OMTensorList creates its own copy of the passed-in tensor pointer
 * array. This copy is freed when the OMTensorList is destroyed.
 *
 * @param tensors array of pointers to OMTensor
 * @param n number of elements in tensors array
 * @param shallow whether we want to perform an OMTensorList shallow destroy,
 * if set to true, the list as well as the pointer to the OMTensor
 * array are freed. The tensors are not destroyed.
 * @return pointer to the OMTensorList created, NULL if creation failed.
 *
 */
OM_EXTERNAL_VISIBILITY OMTensorList *omTensorListCreateWithShallow(
    OMTensor **tensors, int64_t n, int64_t shallow);

/**
 * \brief OMTensorList destroyer
 *
 * Destroy the OMTensorList struct recursively. That is to say, both the
 * ptr to the OMTensor pointers AND the OMTensor pointers are freed.
 *
 * @param list pointer to the OMTensorList to be destroyed.  The function
 * simply returns when pointer is null.
 *
 */
OM_EXTERNAL_VISIBILITY void omTensorListDestroy(OMTensorList *list);

/**
 * \brief OMTensorList shallow destroyer which does not destroy the tensors.
 *
 * It does not destroy the OMTensorList. Rather, both the
 * ptr to the OMTensor pointers AND the OMTensor pointers are freed.
 *
 * @param list pointer to the OMTensorList to be freed. The function
 * simply returns when pointer is null.
 *
 */
OM_EXTERNAL_VISIBILITY void omTensorListDestroyShallow(OMTensorList *list);

/**
 * \brief OMTensorList OMTensor array getter
 *
 * The pointer to OMTensor pointers are returned without copying, so caller
 * should not free the returned pointer.
 *
 * @param list pointer to the OMTensorList
 * @return pointer to the array of OMTensor pointers.
 */
OM_EXTERNAL_VISIBILITY OMTensor **omTensorListGetOmtArray(OMTensorList *list);

/**
 * \brief OMTensorList size getter
 *
 *
 * @param list pointer to the OMTensorList
 * @return number of elements in the OMTensor array.
 */
OM_EXTERNAL_VISIBILITY int64_t omTensorListGetSize(OMTensorList *list);

/**
 * \brief OMTensorList OMTensor getter by index
 *
 * @param list pointer to the OMTensorList
 * @param index index of the OMTensor
 * @return pointer to the OMTensor, NULL if not found.
 */
OM_EXTERNAL_VISIBILITY OMTensor *omTensorListGetOmtByIndex(
    OMTensorList *list, int64_t index);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_OMTENSORLIST_H
