//===------ OnnxMlirInternal.h - Internal OnnxMlir Runtime API Decl -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of internal OMTensor data structures and
// helper functions.
//
//===----------------------------------------------------------------------===//

#ifndef __ONNX_MLIR_INTERNAL_H__
#define __ONNX_MLIR_INTERNAL_H__

#include "OnnxMlirRuntime.h"

#ifdef __cplusplus
#include <cstdint>
#include <string>
#include <vector>

/* ================ Internal C++ API call declaration ================ */

/*----------------------------------------- */
/* C++ API for internal only OMTensor calls */
/*----------------------------------------- */

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
OMTensor *rmrCreateWithShape(std::vector<INDEX_TYPE> dataSizes);

/**
 * OMTensor creator with data sizes, element type and random data
 *
 * @param dataSizes, data sizes array
 * @param lbound (optional), lower bound of the random distribution
 * @param ubound (optional), upper bound of the random distribution
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 * Create a full OMTensor like what rmrCreateWithShape does
 * and also fill the OMTensor data buffer with randomly generated
 * real numbers from a uniform distribution between lbound and ubound.
 */
template <typename T>
OMTensor *rmrCreateWithRandomData(
    std::vector<INDEX_TYPE> dataSizes, T lbound = -1.0, T ubound = 1.0);

/**
 * OMTensor aligned data getter
 *
 * @param rmr, pointer to the OMTensor
 * @return pointer to the aligned data buffer of the OMTensor,
 *         NULL if the aligned data buffer is not set.
 */
void *rmrGetAlignedData(OMTensor *rmr);

/**
 * OMTensor aligned data setter
 *
 * @param rmr, pointer to the OMTensor
 * @param alignedData, aligned data buffer of the OMTensor to be set
 *
 * Set the aligned data buffer pointer of the OMTensor.
 */
void rmrSetAlignedData(OMTensor *rmr, void *alignedData);

/**
 * OMTensor data element getter by offset
 *
 * @param rmr, pointer to the OMTensor
 * @param indexes, multi-dimensional index array of the element
 * @return typed element by reference at the offset computed by the index array.
 */
template <typename T>
T &rmrGetElem(OMTensor *rmr, std::vector<INDEX_TYPE> indexes);

/**
 * OMTensor data element getter by index
 *
 * @param rmr, pointer to the OMTensor
 * @param index, index of the element
 * @return typed element by reference at the linear offset.
 */
template <typename T>
T &rmrGetElemByOffset(OMTensor *rmr, INDEX_TYPE index);

/**
 * OMTensor strides computation
 *
 * @param rmr, pointer to the OMTensor
 * @return data strides of the OMTensor computed from the data sizes.
 */
std::vector<int64_t> rmrComputeStridesFromShape(OMTensor *rmr);

/**
 * OMTensor linear offset computation
 *
 * @param rmr, pointer to the OMTensor
 * @param indexes, multi-dimensional index array
 * @return linear offset.
 */
INDEX_TYPE rmrComputeElemOffset(
    OMTensor *rmr, std::vector<INDEX_TYPE> &indexes);

/**
 * OMTensor index set computation
 *
 * @param rmr, pointer to the OMTensor
 * @return index set (i.e., all valid multi-dimensional array indexes
 *         that can be used to access this OMTensor's constituent elements)
 *         for the whole OMTensor.
 */
std::vector<std::vector<INDEX_TYPE>> rmrComputeIndexSet(OMTensor *rmr);

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
bool rmrAreTwoRmrsClose(
    OMTensor *a, OMTensor *b, float rtol = 1e-5, float atol = 1e-5);

/*---------------------------------------------------- */
/* C++ API for internal only OMTensorList calls */
/*---------------------------------------------------- */

/**
 * OMTensorList creator
 *
 * @return pointer to an empty OMTensorList, NULL if creation failed.
 */
OMTensorList *rmrListCreate(void);

/**
 * OMTensorList OMTensor getter by name
 *
 * @param ormrd, pointer to the OMTensorList
 * @param name, name of the OMTensor
 * @return pointer to the OMTensor, NULL if not found.
 */
OMTensor *rmrListGetRmrByName(OMTensorList *ormrd, std::string name);

#endif
#endif