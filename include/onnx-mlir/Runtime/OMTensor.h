//===-------------- OMTensor.h - OMTensor Declaration header --------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMTensor data structures and
// API functions.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OMTENSOR_H
#define ONNX_MLIR_OMTENSOR_H

#ifdef __cplusplus
#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#else
#include <stdbool.h>
#endif // #ifdef __cplusplus

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif // #ifdef __APPLE__

#include "onnx-mlir/Runtime/OnnxDataType.h"

/* Typically, MemRefs in MLIR context are used as a compile-time constructs.
 * Information such as element type and rank of the data payload is statically
 * encoded, meaning that they are determined and fixed at compile-time. This
 * presents significant burden for any runtime components trying to interact
 * with the compiled executable.
 *
 * Thus a version of MemRef struct that is amenable to runtime manipulation is
 * provided as a basis for building any runtime-related components providing
 * user-facing programming interfaces. All information are dynamically encoded
 * as members of this struct so that they can be accessed and modified easily
 * during runtime.
 *
 * We will refer to it as an OMTensor.
 */

struct OMTensor;

#ifndef __cplusplus
typedef struct OMTensor OMTensor;
#endif

/**
 * \brief Create a OMTensor with specified data pointer, shape, rank and element
 * type.
 *
 * The call will not create a copy of the data. By default, caller is
 * responsible for managing the memory this pointer refers to. Namely, the
 * OMTensor is not the owner of the data. To indicate OMTensor's ownership of
 * data, use `omTensorCreateWithOwnership`. Ownership determines what happens
 * with the OMTensor is destroyed. With ownership of the data, the destruction
 * of the OMTensor will also free the data.
 *
 * @param data_ptr pointer to tensor data. By default, caller is responsible for
 * managing the memory this pointer refers to.
 * @param shape list of integers indicating the tensor shape.
 * @param rank tensor rank.
 * @param dtype tensor element data type.
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 */
OMTensor *omTensorCreate(
    void *data_ptr, int64_t *shape, int64_t rank, OM_DATA_TYPE dtype);

/**
 * \brief Create an OMTensor with specified data pointer, shape, rank and
 * element type, manually setting data ptr ownership.
 *
 * Using this constructor, users can
 * specify whether OMTensor owns the data, which subsequently determines whether
 * the memory space underlying the data will be freed or not when OMTensor gets
 * destroyed.
 *
 * @param data_ptr pointer to tensor data.
 * @param shape list of integers indicating the tensor shape.
 * @param rank tensor rank.
 * @param dtype tensor element data type.
 * @param owning whether OMTensor owns the data, if set to true, OMTensor will
 * release the data_ptr upon destruction.
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 */
OMTensor *omTensorCreateWithOwnership(void *data_ptr, int64_t *shape,
    int64_t rank, OM_DATA_TYPE dtype, int owning);

/**
 * \brief Destroy the OMTensor struct.
 *
 * If OMTensor does not own the data, destroying the omTensor does not free up
 * the memory occupied by the tensor content. If OMTensor owns the data, this
 * function will free up the memory space underlying the tensor as well. The
 * documentation of OMTensor constructors clarifies the ownership semantics.
 *
 * @param tensor pointer to the OMTensor
 *
 */
void omTensorDestroy(OMTensor *tensor);

/**
 * \brief OMTensor data pointer getter.
 *
 * @param tensor pointer to the OMTensor
 * @return pointer to the data buffer of the OMTensor,
 *         NULL if the data buffer is not set.
 */
void *omTensorGetDataPtr(OMTensor *tensor);

/**
 * \brief OMTensor data shape getter.
 *
 * The data shape is returned as a pointer pointing to an array of
 * n 64-bit integers where n is the rank of the tensor.
 *
 * The shape array is returned without copying, so caller should
 * not free the returned pointer.
 *
 * @param tensor pointer to the OMTensor
 * @return pointer to the data shape array.
 */
int64_t *omTensorGetDataShape(OMTensor *tensor);

/**
 * \brief OMTensor data shape setter.
 *
 * n int64 elements are copied from the shape array to indicate the shape of the
 * tensor, where n is the rank of the tensor.
 *
 * The shape array is copied without being freed, so caller is expected to
 * manage the shape array oneself.
 *
 * @param tensor pointer to the OMTensor
 * @param shape data sizes array to be set
 *
 * Set the data sizes array of the OMTensor to the values in the input array.
 */
void omTensorSetShape(OMTensor *tensor, int64_t *shape);

/**
 * \brief OMTensor data strides getter
 *
 * The data strides are returned as a pointer pointing to an array of
 * n 64-bit integers where n is the rank of the tensor.
 *
 * The strides array is returned without copying, so caller should
 * not free the returned pointer.
 *
 * @param tensor pointer to the OMTensor
 * @return pointer to the data strides array.
 */
int64_t *omTensorGetStrides(OMTensor *tensor);

/**
 * \brief OMTensor data strides setter
 *
 * n int64 elements are copied from the strides array to indicate the
 * per-dimension stride of the tensor, where n is the rank of the tensor.
 *
 * The strides array is copied without being freed, so caller is expected to
 * manage the strides array oneself.
 *
 * @param tensor pointer to the OMTensor
 * @param strides tensor strides array to be set.
 *
 * Set the data strides array of the OMTensor to the values in the input array.
 */
void omTensorSetStrides(OMTensor *tensor, int64_t *strides);

/**
 * \brief OMTensor data type getter
 *
 * @param tensor pointer to the OMTensor
 * @return ONNX data type of the data buffer elements.
 */
OM_DATA_TYPE omTensorGetDataType(OMTensor *tensor);

/**
 * \brief OMTensor data type setter
 *
 * @param tensor pointer to the OMTensor
 * @param dataType ONNX data type to be set
 *
 * Set the ONNX data type of the data buffer elements.
 */
void omTensorSetDataType(OMTensor *tensor, OM_DATA_TYPE dataType);

/* Helper function to get the ONNX data type size in bytes */
static inline int getDataTypeSize(OM_DATA_TYPE dataType) {
  return OM_DATA_TYPE_SIZE[dataType];
}

/**
 * \brief OMTensor data buffer size getter
 *
 * @param tensor pointer to the OMTensor
 * @return the total size of the data buffer in bytes.
 */
int64_t omTensorGetDataBufferSize(OMTensor *tensor);

/**
 * \brief OMTensor rank getter
 *
 * @param tensor, pointer to the OMTensor
 * @return rank of data sizes and strides of the OMTensor.
 */
int omTensorGetRank(OMTensor *tensor);

/**
 * \brief OMTensor number of elements getter
 *
 * @param tensor, pointer to the OMTensor
 * @return the number of elements in the data buffer.
 */
int64_t omTensorGetNumElems(OMTensor *tensor);

#endif // ONNX_MLIR_OMTENSOR_H
