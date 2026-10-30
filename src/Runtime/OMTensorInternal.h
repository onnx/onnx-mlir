/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- OMTensorInternal.h - OMTensor Internal Declarations ---------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file declares OMTensor functions that are intended for internal use
// only — by compiler-generated wrapper code, the JNI layer, and other runtime
// implementation files. They are NOT part of the public user-facing API and
// are intentionally omitted from OMTensor.h.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OMTENSOR_INTERNAL_H
#define ONNX_MLIR_OMTENSOR_INTERNAL_H

#include "onnx-mlir/Runtime/OMTensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * OMTensor data shape setter.
 *
 * For internal use only. Shape is fixed at construction time for user-created
 * tensors; this setter exists for compiler-generated wrapper code that
 * constructs tensors from MemRefs after the fact.
 *
 * @param tensor pointer to the OMTensor
 * @param shape data shape array to be copied into the tensor
 */
OM_EXTERNAL_VISIBILITY void omTensorSetShape(
    OMTensor *tensor, const int64_t *shape);

/**
 * OMTensor data strides getter.
 *
 * For internal use only. Strides are derived from shape and are not part of
 * the public API; users should compute strides from the shape if needed.
 *
 * @param tensor pointer to the OMTensor
 * @return pointer to the data strides array.
 */
OM_EXTERNAL_VISIBILITY const int64_t *omTensorGetStrides(
    const OMTensor *tensor);

/**
 * OMTensor data strides setter.
 *
 * For internal use only.
 *
 * @param tensor pointer to the OMTensor
 * @param strides tensor strides array to be copied into the tensor.
 */
OM_EXTERNAL_VISIBILITY void omTensorSetStrides(
    OMTensor *tensor, const int64_t *strides);

/**
 * OMTensor data strides setter with stride values from PyArray strides.
 *
 * For internal use only (Python binding). PyArray stride values are in bytes;
 * they are divided by the element size before being stored.
 *
 * @param tensor pointer to the OMTensor
 * @param stridesInBytes PyArray stride array (bytes per dimension).
 */
OM_EXTERNAL_VISIBILITY void omTensorSetStridesWithPyArrayStrides(
    OMTensor *tensor, const int64_t *stridesInBytes);

/**
 * OMTensor data type setter.
 *
 * For internal use only. The data type is set at construction time; this
 * setter is only called by compiler-generated wrapper code that populates
 * an OMTensor from a MemRef after construction via omTensorCreateUntyped.
 *
 * @param tensor pointer to the OMTensor
 * @param dataType ONNX data type to be set
 */
OM_EXTERNAL_VISIBILITY void omTensorSetDataType(
    OMTensor *tensor, OM_DATA_TYPE dataType);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_OMTENSOR_INTERNAL_H
