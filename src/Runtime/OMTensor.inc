//===--------- OMTensor.inc - C/C++ Neutral OMTensor Implementation--------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of OMTensor data structures
// and helper functions.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#include <cassert>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>
#else
#include <assert.h>
#endif

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <stdio.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"

#ifdef __cplusplus
#include "src/Runtime/OMTensorHelper.h"
#endif

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
    if ((_shape = (int64_t *)malloc(rank * sizeof(int64_t))) &&
        (_stride = (int64_t *)malloc(rank * sizeof(int64_t)))) {
      _allocatedPtr = NULL;
      _alignedPtr = NULL;
      _offset = 0;
      _dataType = ONNX_TYPE_UNDEFINED;
      _rank = rank;
      _owning = false;
    } else {
      if (_shape)
        free(_shape);
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
    if (_owning)
      free(_allocatedPtr);
    free(_shape);
    free(_stride);
  };
#endif
  // Fields are named according to:
  // https://mlir.llvm.org/docs/Dialects/SPIR-V/#lowering-memrefs-to-spvarray-and-spvrtarray

  // On machine without alignment constraints the allocated and aligned pointers
  // are the same. However, on machines with alignment constraints not supported
  // by the memory allocation system, the allocated ptr points to the chunk of
  // memory that is allocated, and the aligned pointer points to a chunk of
  // memory that is allocated and also satisfy the alignment constraints on the
  // machine. For example, on a machine for which malloc returns chunks aligned
  // at 16 byte boundaries, but where tensor must be allocated at 1K boundaries
  // for performance reason, allocated pointer may return 0x1000f and aligned
  // pointer may return 0x10400.
  void *_allocatedPtr; // data buffer
  void *_alignedPtr;   // aligned data buffer that the omt indexes.

  int64_t _offset;  // offset of 1st element
  int64_t *_shape;  // sizes array
  int64_t *_stride; // strides array
  int64_t _rank;    // rank

  OM_DATA_TYPE _dataType; // ONNX data type

  int _owning; // indicates whether the Omt owns the memory space
               // referenced by _allocatedPtr. Omt struct will release the
               // memory space referred to by _allocatedPtr upon destruction if
               // and only if it owns it.
};

/* Helper function to compute the number of data elements */
static inline int64_t getNumOfElems(int64_t *dataSizes, int rank) {
  int64_t numElem = 1;
  for (int i = 0; i < rank; i++)
    numElem *= dataSizes[i];
  return numElem;
}

// Create a OMTensor.
OMTensor *omTensorCreate(
    void *data_ptr, int64_t *shape, int64_t rank, OM_DATA_TYPE dtype) {
  OMTensor *tensor = (OMTensor *)malloc(sizeof(OMTensor));
  if (!tensor)
    return NULL;
  if ((tensor->_shape = (int64_t *)malloc(rank * sizeof(int64_t))) &&
      (tensor->_stride = (int64_t *)malloc(rank * sizeof(int64_t)))) {
    tensor->_allocatedPtr = data_ptr;
    tensor->_alignedPtr = data_ptr;
    tensor->_rank = rank;
    tensor->_dataType = dtype;
    tensor->_owning = false;
  } else {
    if (tensor->_shape)
      free(tensor->_shape);
    return NULL;
  }

  // Using signed indices helps detect when index falls below 0.
  for (int64_t i = rank - 1; i >= 0; i--) {
    tensor->_shape[i] = shape[i];
    if (i == rank - 1)
      tensor->_stride[i] = 1;
    else
      tensor->_stride[i] = tensor->_stride[i + 1] * tensor->_shape[i + 1];
  }
  return tensor;
}

// Create a OMTensor with specified ownership.
OMTensor *omTensorCreateWithOwnership(void *data_ptr, int64_t *shape,
    int64_t rank, OM_DATA_TYPE dtype, int owning) {
  OMTensor *tensor = omTensorCreate(data_ptr, shape, rank, dtype);
  // If ctor fails, return NULL.
  if (!tensor)
    return NULL;
  tensor->_owning = owning;
  return tensor;
}

/**
 * Create an empty OMTensor with specified rank.
 * This function is intentionally left out from the header because it is only
 * used by the wrapper code we emit around inference function that converts
 * MemRefs to OMTensors for user convenience.
 *
 * This constructor returns a
 * partially filled omTensor; prefer using the new omTensorCreateEmpty()
 * function to fill shape & stride fields automatically.
 *
 * @param rank tensor rank
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 */
OMTensor *omTensorCreateEmptyDeprecated(int rank) {
  OMTensor *omt = (OMTensor *)malloc(sizeof(struct OMTensor));
  if (!omt)
    return NULL;
  if ((omt->_shape = (int64_t *)malloc(rank * sizeof(int64_t))) &&
      (omt->_stride = (int64_t *)malloc(rank * sizeof(int64_t)))) {
    omt->_allocatedPtr = NULL;
    omt->_alignedPtr = NULL;
    omt->_offset = 0;
    omt->_dataType = ONNX_TYPE_UNDEFINED;
    omt->_rank = rank;
    omt->_owning = false;
  } else {
    if (omt->_shape)
      free(omt->_shape);
    return NULL;
  }

  return omt;
}

/**
 * Create an OMTensor with the specified shape, rank and element type,
 * allocate uninitialized data for the specified shape.
 * This function is intentionally left out from the header because it is only
 * used by the wrapper code we emit around inference function that converts
 * MemRefs to OMTensors for user convenience.
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
OMTensor *omTensorCreateEmpty(
    int64_t *shape, int64_t rank, OM_DATA_TYPE dtype) {
  OMTensor *tensor =
      omTensorCreateWithOwnership(NULL, shape, rank, dtype, /*owning=*/true);
  // If ctor fails, return null.
  if (!tensor)
    return NULL;
  void *dataPtr = malloc(omTensorGetNumElems(tensor) * getDataTypeSize(dtype));
  if (!dataPtr)
    return NULL;
  tensor->_alignedPtr = dataPtr;
  tensor->_allocatedPtr = dataPtr;
  return tensor;
}

/* OMTensor destroyer */
void omTensorDestroy(OMTensor *tensor) {
  if (tensor->_owning) {
    free(tensor->_allocatedPtr);
    tensor->_allocatedPtr = NULL;
    tensor->_alignedPtr = NULL;
  }
  free(tensor);
}

/* OMTensor data getter */
void *omTensorGetDataPtr(OMTensor *tensor) { return tensor->_alignedPtr; }

/**
 * OMTensor allocated and aligned pointer setter.
 * This function is intentionally left out from the header because it is only
 * used by the wrapper code we emit around inference function that converts
 * MemRefs to OMTensors for user convenience.
 *
 * @param tensor pointer to the OMTensor
 * @param owning whether allocatedPtr should be freed after tensor is destroyed.
 * @param allocatedPtr allocated pointer to tensor content.
 * @param alignedPtr aligned pointer to tensor content. If NULL will be set to
 * allocatedPtr.
 *
 */
void omTensorSetDataPtr(
    OMTensor *tensor, int owning, void *allocatedPtr, void *alignedPtr) {
  if (tensor->_owning) {
    /* If we own the allocated buffer, free it first. */
    free(tensor->_allocatedPtr);
  }
  tensor->_owning = owning;
  tensor->_allocatedPtr = allocatedPtr;
  if (alignedPtr)
    tensor->_alignedPtr = alignedPtr;
  else
    tensor->_alignedPtr = allocatedPtr;
}

/* OMTensor data sizes getter */
int64_t *omTensorGetDataShape(OMTensor *tensor) { return tensor->_shape; }

/* OMTensor data sizes setter */
void omTensorSetShape(OMTensor *tensor, int64_t *shape) {
  for (int i = 0; i < tensor->_rank; i++)
    tensor->_shape[i] = shape[i];
}

/* OMTensor data strides getter */
int64_t *omTensorGetStrides(OMTensor *tensor) { return tensor->_stride; }

/* OMTensor data strides setter */
void omTensorSetStrides(OMTensor *tensor, int64_t *strides) {
  for (int i = 0; i < tensor->_rank; i++)
    tensor->_stride[i] = strides[i];
}

/* OMTensor data type getter */
OM_DATA_TYPE omTensorGetDataType(OMTensor *tensor) { return tensor->_dataType; }

/* OMTensor data type setter */
void omTensorSetDataType(OMTensor *tensor, OM_DATA_TYPE dataType) {
  tensor->_dataType = dataType;
}

/* OMTensor data buffer size getter */
int64_t omTensorGetDataBufferSize(OMTensor *tensor) {
  return getNumOfElems(tensor->_shape, tensor->_rank) *
         getDataTypeSize(tensor->_dataType);
}

/* OMTensor rank getter */
int omTensorGetRank(OMTensor *tensor) { return tensor->_rank; }

/* OMTensor number of elements getter */
int64_t omTensorGetNumElems(OMTensor *tensor) {
  // Using signed indices helps detect when index falls below 0.
  // Verify that strides are dense, meaning that there're
  // no skipping elements.
  for (int64_t i = tensor->_rank - 1; i >= 0; i--) {
    int64_t strideIfNotSkipping = 1;
    for (int64_t j = i + 1; j < tensor->_rank; j++) {
      strideIfNotSkipping *= tensor->_shape[j];
    }
    assert(tensor->_stride[i] == strideIfNotSkipping);
  }
  return getNumOfElems(tensor->_shape, tensor->_rank);
}

/**
 * OMTensor allocated ptr getter.
 * Note that this function is intentionally left out from the header
 * because it is only used by the wrapper code we emit around inference
 * function that converts OMTensor into MemRefs for user convenience.
 *
 * @param tensor pointer to the OMTensor
 * @return pointer to the allocated data buffer of the OMTensor,
 *         NULL if the allocated data buffer is not set.
 */
void *omTensorGetAllocatedPtr(OMTensor *tensor) {
  return tensor->_allocatedPtr;
}

#ifdef __cplusplus
/* OMTensor creator with data sizes and element type  */
template <typename T>
OMTensor *omTensorCreateWithShape(std::vector<int64_t> dataSizes) {
  /* Create a OMTensor with data sizes and strides allocated */
  auto omt = omTensorCreateEmptyDeprecated(dataSizes.size());
  if (omt == NULL)
    return NULL;

  /* Allocate data buffer */
  omt->_rank = dataSizes.size();
  if ((omt->_allocatedPtr = malloc(
           getNumOfElems(dataSizes.data(), omt->_rank) * sizeof(T))) == NULL) {
    omTensorDestroy(omt);
    return NULL;
  }

  omt->_alignedPtr = omt->_allocatedPtr;
  omt->_offset = 0;

  /* Copy dataSizes, _shape already allocated by omTensorCreate */
  copy(dataSizes.begin(), dataSizes.end(), omt->_shape);

  /* Compute and copy dataStrides, _stride already allocated by
   * omTensorCreateEmptyDeprecated
   */
  auto computedStrides = computeStridesFromSizes(omt->_shape, omt->_rank);
  copy(computedStrides.begin(), computedStrides.end(), omt->_stride);

  /* Convert CPP type to ONNX type */
  try {
    omt->_dataType = OM_DATA_TYPE_CPP_TO_ONNX.at(std::string(typeid(T).name()));
  } catch (const std::out_of_range &e) {
    omt->_dataType = ONNX_TYPE_UNDEFINED;
  }

  /* Set flag for destructor */
  omt->_owning = true;

  return omt;
}

/* OMTensor creator with data sizes, element type and random data */
template <typename T>
OMTensor *omTensorCreateWithRandomData(
    std::vector<int64_t> dataSizes, T lbound, T ubound) {
  // Will be used to obtain a seed for the random number engine
  std::random_device rd;
  // Standard mersenne_twister_engine seeded with rd()
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(lbound, ubound);

  auto omt = omTensorCreateWithShape<T>(dataSizes);
  if (omt == NULL)
    return NULL;

  std::generate((T *)omt->_allocatedPtr,
      (T *)omt->_allocatedPtr + getNumOfElems(omt->_shape, omt->_rank),
      [&]() { return dis(gen); });
  return omt;
}

/* Access an element (by reference) at offset computed by index array */
template <typename T>
T &omTensorGetElem(OMTensor *omt, std::vector<int64_t> indexes) {
  int64_t elemOffset = omTensorComputeElemOffset(omt, indexes);
  return ((T *)omt->_allocatedPtr)[elemOffset];
}

/* Access an element (by reference) at linear offset */
template <typename T>
T &omTensorGetElemByOffset(OMTensor *omt, int64_t index) {
  return ((T *)omt->_allocatedPtr)[index];
}

/* Compute strides vector from sizes vector */
std::vector<int64_t> omTensorComputeStridesFromShape(OMTensor *omt) {
  return computeStridesFromSizes(omt->_shape, omt->_rank);
}

/* Compute linear element offset from multi-dimensional index array */
int64_t omTensorComputeElemOffset(
    OMTensor *omt, std::vector<int64_t> &indexes) {
  return computeElemOffset(omt->_stride, omt->_rank, indexes);
}

/* Compute index set for the whole OMTensor */
std::vector<std::vector<int64_t>> omTensorComputeIndexSet(OMTensor *omt) {
  // First, we create index set of each dimension separately.
  // i.e., for a tensor/OMT of shape (2, 3), its dimWiseIdxSet will be:
  // {{0,1}, {0,1,2}};
  std::vector<std::vector<int64_t>> dimWiseIdxSet;
  for (auto dimSize :
      std::vector<int64_t>(omt->_shape, omt->_shape + omt->_rank)) {
    std::vector<int64_t> dimIdxSet(dimSize);
    iota(begin(dimIdxSet), end(dimIdxSet), 0);
    dimWiseIdxSet.emplace_back(dimIdxSet);
  }
  // Then, the cartesian product of vectors within dimWiseIdxSet will be the
  // index set for the whole OMT.
  return CartProduct(dimWiseIdxSet);
}

/* Check whether two OMTensor data are "close" to each other */
template <typename T>
inline bool omTensorAreTwoOmtsClose(
    OMTensor *a, OMTensor *b, float rtol, float atol) {

  // Compare shape.
  auto aShape = std::vector<int64_t>(a->_shape, a->_shape + a->_rank);
  auto bShape = std::vector<int64_t>(b->_shape, b->_shape + b->_rank);
  if (aShape != bShape) {
    std::cerr << "Shape mismatch ";
    printVector(aShape, ",", std::cerr);
    std::cerr << " != ";
    printVector(bShape, ",", std::cerr);
    return false;
  }

  // Compute absolute difference, verify it's within tolerable range.
  auto anum = omTensorGetNumElems(a);
  std::vector<T> absoluteDiff(anum);
  std::transform((T *)a->_allocatedPtr, (T *)a->_allocatedPtr + anum,
      (T *)b->_allocatedPtr, absoluteDiff.begin(), std::minus<>());
  std::transform(absoluteDiff.begin(), absoluteDiff.end(), absoluteDiff.begin(),
      static_cast<T (*)(T)>(&std::abs));
  bool atolSatisfied = std::all_of(
      absoluteDiff.begin(), absoluteDiff.end(), [&](T a) { return a < atol; });

  // Compute relative difference, verify it's within tolerable range.
  std::vector<T> relativeDiff(anum);
  std::transform(absoluteDiff.begin(), absoluteDiff.end(),
      (T *)a->_allocatedPtr, relativeDiff.begin(), std::divides<>());
  bool rtolSatisfied = all_of(
      relativeDiff.begin(), relativeDiff.end(), [&](T a) { return a < rtol; });

  if (atolSatisfied && rtolSatisfied) {
    return true;
  } else {
    // Figure out where and what went wrong, this can be slow; but hopefully we
    // don't need this often.
    for (const auto &idx : omTensorComputeIndexSet(a)) {
      T aElem = omTensorGetElem<T>(a, idx);
      T bElem = omTensorGetElem<T>(b, idx);
      auto elmAbsDiff = abs(aElem - bElem);
      auto withinRtol = (elmAbsDiff / aElem < rtol);
      auto withinAtol = (elmAbsDiff < atol);
      if (!withinRtol || !withinAtol) {
        std::cerr << "a[";
        printVector(idx, ",", std::cerr);
        std::cerr << "] = " << aElem << " != ";
        std::cerr << "b[";
        printVector(idx, ",", std::cerr);
        std::cerr << "] = " << bElem << std::endl;
      }
    }
    return false;
  }
}

// Explicit instantiation of all templated API functions.

template OMTensor *omTensorCreateWithShape<int32_t>(
    std::vector<int64_t> dataSizes);
template OMTensor *omTensorCreateWithShape<int64_t>(
    std::vector<int64_t> dataSizes);
template OMTensor *omTensorCreateWithShape<float>(
    std::vector<int64_t> dataSizes);
template OMTensor *omTensorCreateWithShape<double>(
    std::vector<int64_t> dataSizes);

template OMTensor *omTensorCreateWithRandomData<int32_t>(
    std::vector<int64_t> dataSizes, int32_t lbound, int32_t ubound);
template OMTensor *omTensorCreateWithRandomData<int64_t>(
    std::vector<int64_t> dataSizes, int64_t lbound, int64_t ubound);
template OMTensor *omTensorCreateWithRandomData<float>(
    std::vector<int64_t> dataSizes, float lbound, float ubound);
template OMTensor *omTensorCreateWithRandomData<double>(
    std::vector<int64_t> dataSizes, double lbound, double ubound);

template int32_t &omTensorGetElem<int32_t>(
    OMTensor *, std::vector<int64_t> indexes);
template int64_t &omTensorGetElem<int64_t>(
    OMTensor *, std::vector<int64_t> indexes);
template float &omTensorGetElem<float>(
    OMTensor *, std::vector<int64_t> indexes);
template double &omTensorGetElem<double>(
    OMTensor *, std::vector<int64_t> indexes);

template int32_t &omTensorGetElemByOffset<int32_t>(OMTensor *, int64_t index);
template int64_t &omTensorGetElemByOffset<int64_t>(OMTensor *, int64_t index);
template float &omTensorGetElemByOffset<float>(OMTensor *, int64_t indexs);
template double &omTensorGetElemByOffset<double>(OMTensor *, int64_t index);

template bool omTensorAreTwoOmtsClose<int32_t>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
template bool omTensorAreTwoOmtsClose<int64_t>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
template bool omTensorAreTwoOmtsClose<float>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
template bool omTensorAreTwoOmtsClose<double>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
#endif
