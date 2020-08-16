//===-------- RtMemRef.h - external RtMemRef C/C++ API call header --------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of external RtMemRef data structures and
// helper functions.
//
//===----------------------------------------------------------------------===//
#ifndef __RTMEMREF_H__
#define __RTMEMREF_H__

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

typedef int64_t INDEX_TYPE;

struct RtMemRef;
typedef struct RtMemRef RtMemRef;

struct RtMemRefList;
typedef struct RtMemRefList OrderedRtMemRefDict;

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------- */
/* C/C++ API for RtMemRef calls */
/*----------------------------- */

/**
 * RtMemRef creator
 *
 * @param rank, rank of the data sizes and strides
 * @return pointer to RtMemRef created, NULL if creation failed.
 *
 * Create Create a RtMemRef with specified rank. Memory for data sizes and
 * strides are allocated.
 */
RtMemRef *rmr_create(int rank);

/**
 * RtMemRef destroyer
 *
 * @param rmr, pointer to the RtMemRef
 *
 * Destroy the RtMemRef struct.
 */
void rmr_destroy(RtMemRef *rmr);

/**
 * RtMemRef data getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the data buffer of the RtMemRef,
 *         NULL if the data buffer is not set.
 */
void *rmr_getData(RtMemRef *rmr);

/**
 * RtMemRef data setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param data, data buffer of the RtMemRef to be set
 *
 * Set the data buffer pointer of the RtMemRef. Note that the data buffer
 * is assumed to be managed by the user, i.e., the RtMemRef destructor
 * will not free the data buffer. Because we don't know how exactly the
 * data buffer is allocated, e.g., it could have been allocated on the stack.
 */
void rmr_setData(RtMemRef *rmr, void *data);

/**
 * RtMemRef data sizes getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the data sizes array.
 */
INDEX_TYPE *rmr_getDataSizes(RtMemRef *rmr);

/**
 * RtMemRef data sizes setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param dataSizes, data sizes array to be set
 *
 * Set the data sizes array of the RtMemRef to the values in the input array.
 */
void rmr_setDataSizes(RtMemRef *rmr, INDEX_TYPE *dataSizes);

/**
 * RtMemRef data strides getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the data strides array.
 */
int64_t *rmr_getDataStrides(RtMemRef *rmr);

/**
 * RtMemRef data strides setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param dataStrides, data strides array to be set
 *
 * Set the data strides array of the RtMemRef to the values in the input array.
 */
void rmr_setDataStrides(RtMemRef *rmr, int64_t *dataStrides);

/**
 * RtMemRef data type getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return ONNX data type of the data buffer elements.
 */
int rmr_getDataType(RtMemRef *rmr);

/**
 * RtMemRef data type setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param dataType, ONNX data type to be set
 *
 * Set the ONNX data type of the data buffer elements.
 */
void rmr_setDataType(RtMemRef *rmr, int dataType);

/**
 * RtMemRef data buffer size getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return the total size of the data buffer in bytes.
 */
int64_t rmr_getDataBufferSize(RtMemRef *rmr);

/**
 * RtMemRef rank getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return rank of data sizes and strides of the RtMemRef.
 */
int rmr_getRank(RtMemRef *rmr);

/**
 * RtMemRef name getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the name of the RtMemRef,
 *         an empty string if the name is not set.
 */
char *rmr_getName(RtMemRef *rmr);

/**
 * RtMemRef name setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param name, name of the RtMemRef to be set
 *
 * Set the name of the RtMemRef.
 */
void rmr_setName(RtMemRef *rmr, char *name);

/**
 * RtMemRef number of elements getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return the number of elements in the data buffer.
 */
INDEX_TYPE rmr_getNumOfElems(RtMemRef *rmr);

/*---------------------------------------- */
/* C/C++ API for RtMemRefList calls */
/*---------------------------------------- */

/**
 * RtMemRefList creator
 *
 * @param rmrs, array of pointers to RtMemRef
 * @param n, number of elements in rmrs array
 * @return pointer to the RtMemRefList created, NULL if creation failed.
 *
 * Create an RtMemRefList with specified RtMemRef array.
 * If a RtMemRef has a name, in addition to be accessed by its index,
 * the RtMemRef can also be accessed by its name.
 */
RtMemRefList *ormrd_create(RtMemRef *rmrs[], int n);

/**
 * RtMemRefList destroyer
 *
 * @param ormrd, pointer to the RtMemRefList to be destroyed
 *
 * Destroy the RtMemRefList struct.
 */
void ormrd_destroy(RtMemRefList *ormrd);

/**
 * RtMemRefList RtMemRef array getter
 *
 * @param ormrd, pointer to the RtMemRefList
 * @return pointer to the array of RtMemRef pointers.
 */
RtMemRef **ormrd_getRmrs(RtMemRefList *ormrd);

/**
 * RtMemRefList number of RtMemRefs getter
 *
 * @param ormrd, pointer to the RtMemRefList
 * @return number of elements in the RtMemRef array.
 */
int ormrd_getNumOfRmrs(RtMemRefList *ormrd);

#ifdef __cplusplus
}
#endif

#endif /* __RTMEMREF_H__ */
