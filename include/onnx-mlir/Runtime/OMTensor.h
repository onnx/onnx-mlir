/*
 * SPDX-License-Identifier: Apache-2.0
 */

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

#if defined(__APPLE__) || defined(__MVS__)
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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Create a OMTensor with specified data pointer, shape, rank and element
 * type.
 *
 * The call will create a copy of the shape array but will not create a
 * copy of the data numerical values. The shape array is copied without being freed,
 * so users are expected to manage the shape array oneself.  By default, users are
 * responsible for managing the memory the numerical data pointer refers to, keeping
 * the numerical data alive for the duration of the usage of the created tensor and
 * freeing the numerical data after the last use of the created tensor.  Namely,
 * the OMTensor is not the owner of the numerical data.  To indicate OMTensor's
 * ownership of numerical data, use `omTensorCreateWithOwnership`.  Ownership
 * determines what happens with the OMTensor is destroyed.  With ownership of the
 * numerical data, the destruction of the OMTensor will also free the numerical data
 * associated with the tensor.
 *
 * OM_DATA_TYPE enumerates the ONNX data types: INT/UINT 8/16/32/64, BOOL,
 * FLOAT16, BFLOAT16, FLOAT, DOUBLE, COMPLEX 64/128, and STRING. For example,
 * a 32-bit signed-integer is represented by the ONNX_TYPE_INT32 enum value.
 *
 * @param data_ptr pointer to tensor data numerical values. By default, caller
 * is responsible for managing the memory this pointer refers to.
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
 * The call will create a copy of the shape array but will not create a
 * copy of the numerical data.  The shape array is copied without being freed,
 * so users are expected to manage the shape array oneself. Users can specify
 * whether OMTensor owns the numerical data, which subsequently determines whether
 * the memory space underlying the numerical data will be freed or not when OMTensor
 * gets destroyed. Namely, if the ownership flag is set to false, users are responsible
 * for keeping the numerical data live until the last use of the tensor and freeing the
 * numerical data memory after its last use. If the ownership is set to true, then
 * the destruction of the tensor will also free the numerical data associated with
 * the tensor.
 *
 * OM_DATA_TYPE enumerates the ONNX data types: INT/UINT 8/16/32/64, BOOL,
 * FLOAT16, BFLOAT16, FLOAT, DOUBLE, COMPLEX 64/128, and STRING. For example,
 * a 32-bit unsigned-integer is represented by the ONNX_TYPE_UINT32 enum value.
 *
 * @param data_ptr pointer to tensor numerical data values.
 * @param shape list of integers indicating the tensor shape.
 * @param rank tensor rank.
 * @param dtype tensor element data type.
 * @param owning whether OMTensor owns the data, if set to true, OMTensor will
 * release the data_ptr upon destruction.
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 */
OMTensor *omTensorCreateWithOwnership(void *data_ptr, int64_t *shape,
    int64_t rank, OM_DATA_TYPE dtype, int64_t owning);

/**
 * Create an OMTensor with the specified shape, rank and element type,
 * allocate uninitialized data for the specified shape.
 *
 * The OMTensor created using this constructor owns the underlying memory
 * space allocated to hold the content of the tensor numerical values.
 *
 * OM_DATA_TYPE enumerates the ONNX data types: INT/UINT 8/16/32/64, BOOL,
 * FLOAT16, BFLOAT16, FLOAT, DOUBLE, COMPLEX 64/128, and STRING. For example,
 * a 32-bit floating-point is represented by the ONNX_TYPE_FLOAT enum value.
 *
 * @param shape list of integers indicating the tensor shape.
 * @param rank tensor rank.
 * @param dtype tensor element data type.
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 */
OMTensor *omTensorCreateEmpty(int64_t *shape, int64_t rank, OM_DATA_TYPE dtype);

/**
 * \brief Destroy the OMTensor struct.
 *
 * If OMTensor does not own its numerical data, destroying the omTensor does
 * not free up the memory occupied by the tensor numerical values. If OMTensor
 * owns the numerical data, this function will free up the memory space
 * underlying the tensor's numerical data as well. The documentation of
 * OMTensor constructors clarifies the ownership semantics.
 *
 * @param tensor pointer to the OMTensor. The function simply returns when
 * pointer is null.
 *
 */
void omTensorDestroy(OMTensor *tensor);

/**
 * \brief OMTensor data pointer getter.
 *
 * @param tensor pointer to the OMTensor
 * @return pointer to the numerical data buffer of the OMTensor,
 *         NULL if the numerical data buffer is not set.
 */
void *omTensorGetDataPtr(const OMTensor *tensor);

/**
 * \brief OMTensor data shape getter.
 *
 * The numerical data shape is returned as a pointer pointing to an array of
 * n 64-bit integers where n is the rank of the tensor.
 *
 * The shape array is returned without copying, so caller should
 * not free the returned pointer.
 *
 * @param tensor pointer to the OMTensor
 * @return pointer to the data shape array.
 */
int64_t *omTensorGetShape(const OMTensor *tensor);

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
 * @param shape data shape array to be set
 *
 * Set the data shape array of the OMTensor to the values in the input array.
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
int64_t *omTensorGetStrides(const OMTensor *tensor);

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
void omTensorSetStrides(OMTensor *tensor, int64_t *stride);

/**
 * \brief OMTensor data strides setter with stride values from PyArray strides
 *
 * Note that PyArray stride values are in bytes, while OMTensor stride values in
 * elements. Thus, PyArray stride values will be divided by datatype size before
 * passing to OMTensor stride values.
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
void omTensorSetStridesWithPyArrayStrides(
    OMTensor *tensor, int64_t *stridesInBytes);

/**
 * \brief OMTensor data type getter
 *
 * OM_DATA_TYPE enumerates the ONNX data types: INT/UINT 8/16/32/64, BOOL,
 * FLOAT16, BFLOAT16, FLOAT, DOUBLE, COMPLEX 64/128, and STRING. For example,
 * a string is represented by the ONNX_TYPE_STRING enum value.
 *
 * @param tensor pointer to the OMTensor
 * @return ONNX data type of the data buffer elements.
 */
OM_DATA_TYPE omTensorGetDataType(const OMTensor *tensor);

/**
 * \brief OMTensor data type setter
 *
 * OM_DATA_TYPE enumerates the ONNX data types: INT/UINT 8/16/32/64, BOOL,
 * FLOAT16, BFLOAT16, FLOAT, DOUBLE, COMPLEX 64/128, and STRING. For example,
 * a 1-bit boolean number is represented by the ONNX_TYPE_BOOL enum value.
 *
 * @param tensor pointer to the OMTensor
 * @param dataType ONNX data type to be set
 *
 * Set the ONNX data type of the data buffer elements.
 */
void omTensorSetDataType(OMTensor *tensor, OM_DATA_TYPE dataType);

/* Helper function to get the ONNX data type size in bytes */
static inline int64_t getDataTypeSize(OM_DATA_TYPE dataType) {
  return OM_DATA_TYPE_SIZE[dataType];
}

/**
 * \brief OMTensor numerical data buffer size getter
 *
 * @param tensor pointer to the OMTensor
 * @return the total size of the data buffer in bytes.
 */
int64_t omTensorGetBufferSize(const OMTensor *tensor);

/**
 * \brief OMTensor rank getter
 *
 * @param tensor, pointer to the OMTensor
 * @return rank of data shape and strides of the OMTensor.
 */
int64_t omTensorGetRank(const OMTensor *tensor);

/**
 * \brief OMTensor number of elements getter
 *
 * @param tensor, pointer to the OMTensor
 * @return the number of elements in the data buffer.
 */
int64_t omTensorGetNumElems(const OMTensor *tensor);

/**
 * \brief OMTensor owning flag getter
 *
 * @return owning flag of the OMTensor.
 */
int64_t omTensorGetOwning(const OMTensor *tensor);

/**
 * \brief OMTensor owning flag setter
 */
void omTensorSetOwning(OMTensor *tensor, int64_t owning);

/**
 * Print an OMTensor to stdout.
 *
 * @param msg, pointer to descriptive string
 * @param tensor, pointer to the OMTensor to print
 */
void omTensorPrint(const char *msg, const OMTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_OMTENSOR_H
