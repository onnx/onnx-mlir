/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OMTensorHelper.hpp - OMTensor Helper Func header ----------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMTensor C++ helper functions. At some
// point, this file needs to be merged into the OMTensor.h along with other C++
// APIs operating on OMTensor.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_TENSOR_HELPER_H
#define ONNX_MLIR_TENSOR_HELPER_H

#include <string>
#include <vector>

#include "onnx-mlir/Runtime/OnnxDataType.h"

struct OMTensor;

/**
 * OMTensor creator with data sizes and element type
 *
 * @param dataSizes, data sizes array
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 * Create a full OMTensor of data type T and shape dataSizes, with all
 * data fields initialized to proper values and data pointers malloc'ed.
 */
template <typename T>
OMTensor *omTensorCreateWithShape(const std::vector<int64_t> &dataSizes);

/**
 * omDefineSeed.
 * When called, the random number generator for omTensorCreateWithRandomData
 * will be seeded exactly once. The seed is randomly generated when ignoreSeed
 * is nonnull; otherwise the input seedValue is used.
 *
 * @param seed input seed.
 * @param hasSeedValue when nonzero, this function uses the provided seed.
 * Otherwise, this function defines its own random seed.
 * @return return the seed that was used.
 *
 */
unsigned int omDefineSeed(unsigned int seed, unsigned int hasSeedValue);

/**
 * Create an OMTensor filled with random data for a given OM_DATA_TYPE.
 *
 * Accepts the element type as a runtime OM_DATA_TYPE value, making it possible
 * to create randomly-filled tensors when the type is only known at runtime
 * (e.g. when driving a model from its input signature).
 *
 * Type-specific behaviour:
 *   - Numeric types: uniform distribution over [lbound, ubound].
 *   - ONNX_TYPE_BOOL: lbound/ubound are clamped to {false, true} (threshold
 *     0.5); default [0, 1] gives equal probability of false and true.
 *   - ONNX_TYPE_FLOAT16: lbound/ubound are cast to float, converted via
 *     om_f32_to_f16, and stored as uint16_t raw bits.
 *   - ONNX_TYPE_STRING: lbound/ubound are cast to int and used as the range
 *     for random integers whose decimal representations become the elements.
 *     Default bounds [0, 63] can be overridden like any numeric type.
 *     The tensor data buffer uses the packed layout of
 *     omTensorBuildStringBuffer: [ptr0..ptrN-1][str0\0 str1\0 ... strN-1\0].
 *   - Complex and sub-byte types (float8, int4, uint4): unsupported and will
 *     return NULL.
 *
 * @param shape   shape of the tensor to create.
 * @param omType  element type expressed as an OM_DATA_TYPE enum value.
 * @param lbound  lower bound of the random distribution (default -1.0).
 * @param ubound  upper bound of the random distribution (default  1.0).
 * @return pointer to the OMTensor created, or NULL on failure.
 */
OMTensor *omTensorCreateWithRandomData(const std::vector<int64_t> &shape,
    OM_DATA_TYPE omType, double lbound = -1.0, double ubound = 1.0);

/**
 * Build the flat pointer+data buffer required by ONNX_TYPE_STRING OMTensors.
 *
 * ONNX string tensors are not stored as an array of std::string.  Instead, the
 * OMTensor data pointer must point to a single contiguous allocation with this
 * layout:
 *
 *   [ptr0][ptr1]...[ptrN-1][str0\0][str1\0]...[strN-1\0]
 *    <---- N char* pointers ---->  <----- packed string data ---->
 *
 * Each ptrK points into the string-data region of the same buffer.  Callers
 * read elements as ((const char **)dataPtr)[k].
 *
 * Use this helper whenever you need to create or populate a string OMTensor
 * from a std::vector<std::string>.  The returned buffer is a single malloc
 * allocation; pass it (with owning=1) to omTensorCreateWithOwnership so the
 * tensor destructor frees it correctly.
 *
 * @param strings  string values to pack, one per tensor element.
 * @return malloc'd buffer in the layout above, or nullptr on failure.
 */
void *omTensorBuildStringBuffer(const std::vector<std::string> &strings);

/**
 * Create an OMTensor with a "sequence of ones followed by zeros" (soz) fill
 * along the innermost dimension.
 *
 * For each row (product of all but the last dimension), the first sozCount
 * elements are set to 1 and the remaining elements to 0.  Well suited for
 * sequence-length masks and attention-mask inputs.
 *
 * @param shape    shape of the tensor; must have rank >= 1.
 * @param omType   element type (any numeric type supporting 0 and 1).
 * @param sozCount number of leading ones per row (>= 0), or -1 to choose
 *                 independently and randomly for each row in [1, innerDim-1].
 *                 Values >= innerDim are capped at innerDim (all ones).
 * @return pointer to the OMTensor created, or nullptr on failure.
 */
OMTensor *omTensorCreateSozData(
    const std::vector<int64_t> &shape, OM_DATA_TYPE omType, int64_t sozCount);

/**
 * OMTensor data element getter by offset
 *
 * @param omt, pointer to the OMTensor
 * @param indexes, multi-dimensional index array of the element
 * @return typed element by reference at the offset computed by the index array.
 */
template <typename T>
T &omTensorGetElem(const OMTensor *omt, const std::vector<int64_t> &indexes);

/**
 * OMTensor data element getter by index
 *
 * @param omt, pointer to the OMTensor
 * @param index, index of the element
 * @return typed element by reference at the linear offset.
 */
template <typename T>
T &omTensorGetElemByOffset(const OMTensor *omt, int64_t index);

/**
 * OMTensor strides computation
 *
 * @param omt, pointer to the OMTensor
 * @return data strides of the OMTensor computed from the data sizes.
 */
std::vector<int64_t> omTensorComputeStridesFromShape(const OMTensor *omt);

/**
 * OMTensor linear offset computation
 *
 * @param omt, pointer to the OMTensor
 * @param indexes, multi-dimensional index array
 * @return linear offset.
 */
int64_t omTensorComputeElemOffset(
    const OMTensor *omt, const std::vector<int64_t> &indexes);

/**
 * OMTensor index set computation
 *
 * @param omt, pointer to the OMTensor
 * @return index set (i.e., all valid multi-dimensional array indexes
 *         that can be used to access this OMTensor's constituent elements)
 *         for the whole OMTensor.
 */
std::vector<std::vector<int64_t>> omTensorComputeIndexSet(const OMTensor *omt);

/**
 * OMTensor "distance" computation
 *
 * @param a, 1st OMTensor
 * @param b, 2nd OMTensor
 * @param rtol (optional), relative difference tolerance
 * @param atol (optional), absolute difference tolerance
 * @return true if both relative and absolute difference are within the
 *         specified tolerance, respectively, false otherwise.
 */
template <typename T>
bool omTensorAreTwoOmtsClose(
    const OMTensor *a, const OMTensor *b, float rtol = 1e-5, float atol = 1e-5);
#endif