//===-------- _RtMemRef.h - Internal RtMemRef C++ API call header --------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of internal RtMemRef data structures and
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
/* C++ API for internal only RtMemRef calls */
/*----------------------------------------- */

/**
 * RtMemRef creator with data sizes and element type
 *
 * @param dataSizes, data sizes array
 * @return pointer to RtMemRef created, NULL if creation failed.
 *
 * Create a full RtMemRef of data type T and shape dataSizes, with all
 * data fields initialized to proper values and data pointers malloc'ed.
 */
template <typename T>
RtMemRef *rmrCreateWithShape(std::vector<INDEX_TYPE> dataSizes);

/**
 * RtMemRef creator with data sizes, element type and random data
 *
 * @param dataSizes, data sizes array
 * @param lbound (optional), lower bound of the random distribution
 * @param ubound (optional), upper bound of the random distribution
 * @return pointer to RtMemRef created, NULL if creation failed.
 *
 * Create a full RtMemRef like what rmrCreateWithShape does
 * and also fill the RtMemRef data buffer with randomly generated
 * real numbers from a uniform distribution between lbound and ubound.
 */
template <typename T>
RtMemRef *rmrCreateWithRandomData(
    std::vector<INDEX_TYPE> dataSizes, T lbound = -1.0, T ubound = 1.0);

/**
 * RtMemRef aligned data getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the aligned data buffer of the RtMemRef,
 *         NULL if the aligned data buffer is not set.
 */
void *rmrGetAlignedData(RtMemRef *rmr);

/**
 * RtMemRef aligned data setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param alignedData, aligned data buffer of the RtMemRef to be set
 *
 * Set the aligned data buffer pointer of the RtMemRef.
 */
void rmrSetAlignedData(RtMemRef *rmr, void *alignedData);

/**
 * RtMemRef data element getter by offset
 *
 * @param rmr, pointer to the RtMemRef
 * @param indexes, multi-dimensional index array of the element
 * @return typed element by reference at the offset computed by the index array.
 */
template <typename T>
T &rmrGetElem(RtMemRef *rmr, std::vector<INDEX_TYPE> indexes);

/**
 * RtMemRef data element getter by index
 *
 * @param rmr, pointer to the RtMemRef
 * @param index, index of the element
 * @return typed element by reference at the linear offset.
 */
template <typename T>
T &rmrGetElemByOffset(RtMemRef *rmr, INDEX_TYPE index);

/**
 * RtMemRef strides computation
 *
 * @param rmr, pointer to the RtMemRef
 * @return data strides of the RtMemRef computed from the data sizes.
 */
std::vector<int64_t> rmrComputeStridesFromShape(RtMemRef *rmr);

/**
 * RtMemRef linear offset computation
 *
 * @param rmr, pointer to the RtMemRef
 * @param indexes, multi-dimensional index array
 * @return linear offset.
 */
INDEX_TYPE rmrComputeElemOffset(
    RtMemRef *rmr, std::vector<INDEX_TYPE> &indexes);

/**
 * RtMemRef index set computation
 *
 * @param rmr, pointer to the RtMemRef
 * @return index set (i.e., all valid multi-dimensional array indexes
 *         that can be used to access this RtMemRef's constituent elements)
 *         for the whole RtMemRef.
 */
std::vector<std::vector<INDEX_TYPE>> rmrComputeIndexSet(RtMemRef *rmr);

/**
 * RtMemRef "distance" computation
 *
 * @param a, 1st RtMemRef
 * @param b, 2nd RtMemRef
 * @param rtol (optional), relative difference tolerance
 * @param atol (optional), absolute difference tolerance
 * @return true if both relative and absolute difference are within the
 *         specified tolerance, respectively, false otherwise.
 */
template <typename T>
bool rmrAreTwoRmrsClose(
    RtMemRef *a, RtMemRef *b, float rtol = 1e-5, float atol = 1e-5);

/*---------------------------------------------------- */
/* C++ API for internal only RtMemRefList calls */
/*---------------------------------------------------- */

/**
 * RtMemRefList creator
 *
 * @return pointer to an empty RtMemRefList, NULL if creation failed.
 */
RtMemRefList *rmrListCreate(void);

/**
 * RtMemRefList RtMemRef getter by name
 *
 * @param ormrd, pointer to the RtMemRefList
 * @param name, name of the RtMemRef
 * @return pointer to the RtMemRef, NULL if not found.
 */
RtMemRef *rmrListGetRmrByName(RtMemRefList *ormrd, std::string name);

#endif
#endif