//===-------------- OMTensor.h - OMTensor Declaration header --------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMTensor and data structures and
// helper functions.
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
 * We will refer to it as a RMF (Runtime MemRef).
 */

struct OMTensor {
#ifdef __cplusplus
    /**
     * Constructor
     *
     * @param rank, rank of data sizes and strides
     *
     * Create a OMTensor with specified rank. Memory for data sizes and strides
     * are allocated.
     */
    OMTensor(int rank) {
        if ((_dataSizes = (int64_t *)malloc(rank * sizeof(int64_t))) &&
            (_dataStrides = (int64_t *)malloc(rank * sizeof(int64_t)))) {
            _data = NULL;
            _alignedData = NULL;
            _offset = 0;
            _dataType = ONNX_TYPE_UNDEFINED;
            _rank = rank;
            _owningData = false;
        } else {
            throw std::runtime_error(
                    "OMTensor(" + std::to_string(rank) + ") malloc error");
        }
    };

    OMTensor() = default;

    /**
     * Destructor
     *
     * Destroy the OMTensor struct.
     */
    ~OMTensor() {
        if (_owningData)
            free(_data);
        free(_dataSizes);
        free(_dataStrides);
    };
#endif

    void *_data;            // data buffer
    void *_alignedData;     // aligned data buffer that the omt indexes.
    int64_t _offset;     // offset of 1st element
    int64_t *_dataSizes; // sizes array
    int64_t *_dataStrides;  // strides array
    int _dataType;          // ONNX data type
    int _rank;              // rank
    char *_name;            // optional name for named access
    bool _owningData;       // indicates whether the Omt owns the memory space
    // referenced by _data. Omt struct will release the memory
    // space referred to by _data upon destruction if and only
    // if it owns it.
};

#ifndef __cplusplus
typedef struct OMTensor OMTensor;
#endif

/* Helper function to compute the number of data elements */
static inline int64_t getNumOfElems(int64_t *dataSizes, int rank) {
    int64_t numElem = 1;
    for (int i = 0; i < rank; i++)
        numElem *= dataSizes[i];
    return numElem;
}


/**
 * Create an OMTensor.
 *
 * @param rank, rank of the data sizes and strides
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 * Create a OMTensor with specified rank. Memory for data sizes and
 * strides are allocated.
 */
OMTensor *omTensorCreate(void* data_ptr, size_t* shape, size_t rank, OM_DATA_TYPE dtype);

/**
 * OMTensor creator
 *
 * @param rank, rank of the data sizes and strides
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 * Create a OMTensor with specified rank. Memory for data sizes and
 * strides are allocated.
 */
OMTensor *omTensorCreateEmpty(int rank);

/**
 * OMTensor creator
 *
 * @param rank, rank of the data sizes and strides
 * @param name, (optional) name of the tensor
 * @param owningData, whether the omt owns the underlying data, if true, data
 * pointer will be released when the corresponding omt gets released or goes out
 * of scope.
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 * Create a OMTensor with specified rank, name and data ownership. Memory for
 * data sizes and strides are allocated.
 */
OMTensor *omTensorCreateWithNameAndOwnership(
        int rank, char *name, bool owningData);

/**
 * OMTensor destroyer
 *
 * @param omt, pointer to the OMTensor
 *
 * Destroy the OMTensor struct.
 */
void omTensorDestroy(OMTensor *omt);

/**
 * OMTensor data getter
 *
 * @param omt, pointer to the OMTensor
 * @return pointer to the data buffer of the OMTensor,
 *         NULL if the data buffer is not set.
 */
void *omTensorGetData(OMTensor *omt);

/**
 * OMTensor data setter
 *
 * @param omt, pointer to the OMTensor
 * @param data, data buffer of the OMTensor to be set
 *
 * Set the data buffer pointer of the OMTensor. Note that the data buffer
 * is assumed to be managed by the user, i.e., the OMTensor destructor
 * will not free the data buffer. Because we don't know how exactly the
 * data buffer is allocated, e.g., it could have been allocated on the stack.
 */
void omTensorSetData(OMTensor *omt, void *data);

/**
 * OMTensor data sizes getter
 *
 * @param omt, pointer to the OMTensor
 * @return pointer to the data shape array.
 */
int64_t *omTensorGetDataShape(OMTensor *omt);

/**
 * OMTensor data sizes setter
 *
 * @param omt, pointer to the OMTensor
 * @param dataSizes, data sizes array to be set
 *
 * Set the data sizes array of the OMTensor to the values in the input array.
 */
void omTensorSetDataShape(OMTensor *omt, int64_t *dataSizes);

/**
 * OMTensor data strides getter
 *
 * @param omt, pointer to the OMTensor
 * @return pointer to the data strides array.
 */
int64_t *omTensorGetDataStrides(OMTensor *omt);

/**
 * OMTensor data strides setter
 *
 * @param omt, pointer to the OMTensor
 * @param dataStrides, data strides array to be set
 *
 * Set the data strides array of the OMTensor to the values in the input array.
 */
void omTensorSetDataStrides(OMTensor *omt, int64_t *dataStrides);

/**
 * OMTensor data type getter
 *
 * @param omt, pointer to the OMTensor
 * @return ONNX data type of the data buffer elements.
 */
int omTensorGetDataType(OMTensor *omt);

/**
 * OMTensor data type setter
 *
 * @param omt, pointer to the OMTensor
 * @param dataType, ONNX data type to be set
 *
 * Set the ONNX data type of the data buffer elements.
 */
void omTensorSetDataType(OMTensor *omt, int dataType);

/* Helper function to get the ONNX data type size in bytes */
static inline int getDataTypeSize(int dataType) {
    return dataType < 0 ||
           dataType >= sizeof(OM_DATA_TYPE_SIZE) / sizeof(int)
           ? 0
           : OM_DATA_TYPE_SIZE[dataType];
}

/**
 * OMTensor data buffer size getter
 *
 * @param omt, pointer to the OMTensor
 * @return the total size of the data buffer in bytes.
 */
int64_t omTensorGetDataBufferSize(OMTensor *omt);

/**
 * OMTensor rank getter
 *
 * @param omt, pointer to the OMTensor
 * @return rank of data sizes and strides of the OMTensor.
 */
int omTensorGetRank(OMTensor *omt);

/**
 * OMTensor name getter
 *
 * @param omt, pointer to the OMTensor
 * @return pointer to the name of the OMTensor,
 *         an empty string if the name is not set.
 */
char *omTensorGetName(OMTensor *omt);

/**
 * OMTensor name setter
 *
 * @param omt, pointer to the OMTensor
 * @param name, name of the OMTensor to be set
 *
 * Set the name of the OMTensor.
 */
void omTensorSetName(OMTensor *omt, char *name);

/**
 * OMTensor number of elements getter
 *
 * @param omt, pointer to the OMTensor
 * @return the number of elements in the data buffer.
 */
int64_t omTensorGetNumElems(OMTensor *omt);

/*------------------------------------------------------- */
/* Internal function used by OMTensor itself, not exposed */
/*------------------------------------------------------- */

#ifdef __cplusplus

/* Helper function to compute cartisian product */
static inline std::vector<std::vector<int64_t>> CartProduct(
        const std::vector<std::vector<int64_t>> &v) {
    std::vector<std::vector<int64_t>> s = {{}};
    for (const auto &u : v) {
        std::vector<std::vector<int64_t>> r;
        for (const auto &x : s) {
            for (const auto y : u) {
                r.push_back(x);
                r.back().push_back(y);
            }
        }
        s = move(r);
    }
    return s;
}

/* Helper function to compute data strides from sizes */
static inline std::vector<int64_t> computeStridesFromSizes(
        int64_t *dataSizes, int rank) {
    // Shift dimension sizes one to the left, fill in the vacated rightmost
    // element with 1; this gets us a vector that'll be more useful for computing
    // strides of memory access along each dimension using prefix product (aka
    // partial_sum with a multiply operator below). The intuition is that the size
    // of the leading dimension does not matter when computing strides.
    std::vector<int64_t> sizesVec(dataSizes + 1, dataSizes + rank);
    sizesVec.push_back(1);

    std::vector<int64_t> dimStrides(rank);
    partial_sum(sizesVec.rbegin(), sizesVec.rend(), dimStrides.rbegin(),
                std::multiplies<>());
    return dimStrides;
}

/* Helper function to compute linear offset from a multi-dimensional index array
 */
static inline int64_t computeElemOffset(
        int64_t *dataStrides, int rank, std::vector<int64_t> &indexes) {
    auto dimStrides = std::vector<int64_t>(dataStrides, dataStrides + rank);
    int64_t elemOffset = inner_product(
            indexes.begin(), indexes.end(), dimStrides.begin(), (int64_t)0);
    return elemOffset;
}

/* Helper function to print a vector with delimiter */
template <typename T>
static inline void printVector(std::vector<T> vec, std::string _delimiter = ",",
                               std::ostream &stream = std::cout) {
    std::string delimiter;
    for (const auto &elem : vec) {
        stream << delimiter << elem;
        delimiter = _delimiter;
    }
}

#endif

#endif //ONNX_MLIR_OMTENSOR_H
