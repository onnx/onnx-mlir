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
typedef struct RtMemRefList RtMemRefList;

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
 * Create a RtMemRef with specified rank. Memory for data sizes and
 * strides are allocated.
 */
RtMemRef *rmrCreate(int rank);

/**
 * RtMemRef creator
 *
 * @param rank, rank of the data sizes and strides
 * @param name, (optional) name of the tensor
 * @param owningData, whether the rmr owns the underlying data, if true, data
 * pointer will be released when the corresponding rmr gets released or goes out
 * of scope.
 * @return pointer to RtMemRef created, NULL if creation failed.
 *
 * Create a RtMemRef with specified rank, name and data ownership. Memory for
 * data sizes and strides are allocated.
 */
RtMemRef *rmrCreateWithNameAndOwnership(int rank, char *name, bool owningData);

/**
 * RtMemRef destroyer
 *
 * @param rmr, pointer to the RtMemRef
 *
 * Destroy the RtMemRef struct.
 */
void rmrDestroy(RtMemRef *rmr);

/**
 * RtMemRef data getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the data buffer of the RtMemRef,
 *         NULL if the data buffer is not set.
 */
void *rmrGetData(RtMemRef *rmr);

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
void rmrSetData(RtMemRef *rmr, void *data);

/**
 * RtMemRef data sizes getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the data shape array.
 */
INDEX_TYPE *rmrGetDataShape(RtMemRef *rmr);

/**
 * RtMemRef data sizes setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param dataSizes, data sizes array to be set
 *
 * Set the data sizes array of the RtMemRef to the values in the input array.
 */
void rmrSetDataShape(RtMemRef *rmr, INDEX_TYPE *dataSizes);

/**
 * RtMemRef data strides getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the data strides array.
 */
int64_t *rmrGetDataStrides(RtMemRef *rmr);

/**
 * RtMemRef data strides setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param dataStrides, data strides array to be set
 *
 * Set the data strides array of the RtMemRef to the values in the input array.
 */
void rmrSetDataStrides(RtMemRef *rmr, int64_t *dataStrides);

/**
 * RtMemRef data type getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return ONNX data type of the data buffer elements.
 */
int rmrGetDataType(RtMemRef *rmr);

/**
 * RtMemRef data type setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param dataType, ONNX data type to be set
 *
 * Set the ONNX data type of the data buffer elements.
 */
void rmrSetDataType(RtMemRef *rmr, int dataType);

/**
 * RtMemRef data buffer size getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return the total size of the data buffer in bytes.
 */
int64_t rmrGetDataBufferSize(RtMemRef *rmr);

/**
 * RtMemRef rank getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return rank of data sizes and strides of the RtMemRef.
 */
int rmrGetRank(RtMemRef *rmr);

/**
 * RtMemRef name getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the name of the RtMemRef,
 *         an empty string if the name is not set.
 */
char *rmrGetName(RtMemRef *rmr);

/**
 * RtMemRef name setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param name, name of the RtMemRef to be set
 *
 * Set the name of the RtMemRef.
 */
void rmrSetName(RtMemRef *rmr, char *name);

/**
 * RtMemRef number of elements getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return the number of elements in the data buffer.
 */
INDEX_TYPE rmrGetNumElems(RtMemRef *rmr);

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
RtMemRefList *rmrListCreate(RtMemRef **rmrs, int n);

/**
 * RtMemRefList destroyer
 *
 * @param ormrd, pointer to the RtMemRefList to be destroyed
 *
 * Destroy the RtMemRefList struct.
 */
void rmrListDestroy(RtMemRefList *ormrd);

/**
 * RtMemRefList RtMemRef array getter
 *
 * @param ormrd, pointer to the RtMemRefList
 * @return pointer to the array of RtMemRef pointers.
 */
RtMemRef **rmrListGetPtrToRmrs(RtMemRefList *ormrd);

/**
 * RtMemRefList number of RtMemRefs getter
 *
 * @param ormrd, pointer to the RtMemRefList
 * @return number of elements in the RtMemRef array.
 */
int rmrListGetNumRmrs(RtMemRefList *ormrd);

#ifdef __cplusplus
}
#endif

#endif /* __RTMEMREF_H__ */
