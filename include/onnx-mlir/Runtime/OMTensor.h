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
 * We will refer to it as an OMTensor.
 */

struct OMTensor;

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
 * \brief Create an OMTensor with the specified shape, rank and element type,
 * allocate uninitialized data for the specified shape.
 *
 * The OMTensor created using this constructor owns the underlying memory
 * space allocated to the content of the tensor.
 *
 * @param shape list of integers indicating the tensor shape.
 * @param rank tensor rank.
 * @param dtype tensor element data type.
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 */
OMTensor *omTensorCreateEmpty(int64_t *shape, int64_t rank, OM_DATA_TYPE dtype);

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
 * \brief Create an empty OMTensor with specified rank.
 *
 * This constructor returns a
 * partially filled omTensor; prefer using the new omTensorCreateEmpty()
 * function to fill shape & stride fields automatically.
 *
 * @param rank tensor rank
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 */
OMTensor *omTensorCreateEmptyDeprecated(int rank);

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
 * @param tensor pointer to the OMTensor
 * @return pointer to the data strides array.
 */
int64_t *omTensorGetStrides(OMTensor *tensor);

/**
 * \brief OMTensor data strides setter
 *
 *   n int64 elements are copied from the strides array to indicate the
 * per-dimension stride of the tensor, where n is the rank of the tensor.
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
OMTensor *omTensorCreateWithShape(std::vector<int64_t> dataSizes);

/**
 * OMTensor creator with data sizes, element type and random data
 *
 * @param dataSizes, data sizes array
 * @param lbound (optional), lower bound of the random distribution
 * @param ubound (optional), upper bound of the random distribution
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 * Create a full OMTensor like what omTensorCreateWithShape does
 * and also fill the OMTensor data buffer with randomly generated
 * real numbers from a uniform distribution between lbound and ubound.
 */
template <typename T>
OMTensor *omTensorCreateWithRandomData(
    std::vector<int64_t> dataSizes, T lbound = -1.0, T ubound = 1.0);

/**
 * OMTensor data element getter by offset
 *
 * @param omt, pointer to the OMTensor
 * @param indexes, multi-dimensional index array of the element
 * @return typed element by reference at the offset computed by the index array.
 */
template <typename T>
T &omTensorGetElem(OMTensor *omt, std::vector<int64_t> indexes);

/**
 * OMTensor data element getter by index
 *
 * @param omt, pointer to the OMTensor
 * @param index, index of the element
 * @return typed element by reference at the linear offset.
 */
template <typename T>
T &omTensorGetElemByOffset(OMTensor *omt, int64_t index);

/**
 * OMTensor strides computation
 *
 * @param omt, pointer to the OMTensor
 * @return data strides of the OMTensor computed from the data sizes.
 */
std::vector<int64_t> omTensorComputeStridesFromShape(OMTensor *omt);

/**
 * OMTensor linear offset computation
 *
 * @param omt, pointer to the OMTensor
 * @param indexes, multi-dimensional index array
 * @return linear offset.
 */
int64_t omTensorComputeElemOffset(OMTensor *omt, std::vector<int64_t> &indexes);

/**
 * OMTensor index set computation
 *
 * @param omt, pointer to the OMTensor
 * @return index set (i.e., all valid multi-dimensional array indexes
 *         that can be used to access this OMTensor's constituent elements)
 *         for the whole OMTensor.
 */
std::vector<std::vector<int64_t>> omTensorComputeIndexSet(OMTensor *omt);

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
    OMTensor *a, OMTensor *b, float rtol = 1e-5, float atol = 1e-5);

#endif

#endif // ONNX_MLIR_OMTENSOR_H
