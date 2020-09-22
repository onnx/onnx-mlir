//===------- OMTensor.inc - C/C++ Neutral OMTensor Implementation ----------===//
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
using namespace std;
#endif

/* ================ External C/C++ API call implementation ================ */

/*----------------------------- */
/* C/C++ API for OMTensor calls */
/*----------------------------- */

OMTensor *omTensorCreate(void* data_ptr, size_t* shape, size_t rank, OM_DATA_TYPE dtype) {
    OMTensor *omt = (OMTensor *)malloc(sizeof(struct OMTensor));
    if ((omt->_dataSizes = (int64_t *)malloc(rank * sizeof(int64_t))) &&
        (omt->_dataStrides = (int64_t *)malloc(rank * sizeof(int64_t)))) {
        omt->_data = data_ptr;
        omt->_alignedData = data_ptr;
        omt->_rank = rank;
        omt->_dataType = dtype;
    }
    for (size_t i = rank-1; i>=0; i--) {
        omt->_dataSizes[i] = shape[i];
        int64_t stride = 1;
        for (size_t j = i-1; j>=0; j--)
            stride *= omt->_dataStrides[j];
        omt->_dataStrides[i] = stride;
    }
}

/* OMTensor creator */
OMTensor *omTensorCreateEmpty(int rank) {
  OMTensor *omt = (OMTensor *)malloc(sizeof(struct OMTensor));
  if ((omt->_dataSizes = (int64_t *)malloc(rank * sizeof(int64_t))) &&
      (omt->_dataStrides = (int64_t *)malloc(rank * sizeof(int64_t)))) {
    omt->_data = NULL;
    omt->_alignedData = NULL;
    omt->_offset = 0;
    omt->_dataType = ONNX_TYPE_UNDEFINED;
    omt->_rank = rank;
    omt->_name = NULL;
    omt->_owningData = false;
  }
  return omt;
}

/* More detailed OMTensor creator */
OMTensor *omTensorCreateWithNameAndOwnership(int rank, char *name, bool ownership) {
  OMTensor *omt = (OMTensor *)malloc(sizeof(struct OMTensor));
  if ((omt->_dataSizes = (int64_t *)malloc(rank * sizeof(int64_t))) &&
      (omt->_dataStrides = (int64_t *)malloc(rank * sizeof(int64_t)))) {
    omt->_data = NULL;
    omt->_alignedData = NULL;
    omt->_offset = 0;
    omt->_dataType = ONNX_TYPE_UNDEFINED;
    omt->_rank = rank;
    omt->_name = name;
    omt->_owningData = ownership;
  }
}

/* OMTensor destroyer */
// TODO(tjingrant): properly clean up with respect to ownership.
void omTensorDestroy(OMTensor *omt) { free(omt); }

/* OMTensor data getter */
void *omTensorGetData(OMTensor *omt) { return omt->_data; }

/* OMTensor data setter */
void omTensorSetData(OMTensor *omt, void *data) {
  /* If we own the data buffer, free it first. */
  if (omt->_owningData) {
    free(omt->_data);
  }
  omt->_data = data;
}

/* OMTensor data sizes getter */
int64_t *omTensorGetDataShape(OMTensor *omt) { return omt->_dataSizes; }

/* OMTensor data sizes setter */
void omTensorSetDataShape(OMTensor *omt, int64_t *dataSizes) {
  for (int i = 0; i < omt->_rank; i++)
    omt->_dataSizes[i] = dataSizes[i];
}

/* OMTensor data strides getter */
int64_t *omTensorGetDataStrides(OMTensor *omt) { return omt->_dataStrides; }

/* OMTensor data strides setter */
void omTensorSetDataStrides(OMTensor *omt, int64_t *dataStrides) {
  for (int i = 0; i < omt->_rank; i++)
    omt->_dataStrides[i] = dataStrides[i];
}

/* OMTensor data type getter */
int omTensorGetDataType(OMTensor *omt) { return omt->_dataType; }

/* OMTensor data type setter */
void omTensorSetDataType(OMTensor *omt, int dataType) {
  omt->_dataType = dataType;
}

/* OMTensor data buffer size getter */
int64_t omTensorGetDataBufferSize(OMTensor *omt) {
  return getNumOfElems(omt->_dataSizes, omt->_rank) *
         getDataTypeSize(omt->_dataType);
}

/* OMTensor rank getter */
int omTensorGetRank(OMTensor *omt) { return omt->_rank; }

/* OMTensor name getter */
char *omTensorGetName(OMTensor *omt) { return omt->_name; }

/* OMTensor name setter */
void omTensorSetName(OMTensor *omt, char *name) { omt->_name = name; }

/* OMTensor number of elements getter */
int64_t omTensorGetNumElems(OMTensor *omt) {
  return getNumOfElems(omt->_dataSizes, omt->_rank);
}

/* ================ C++ API call implementation ================ */
#ifdef __cplusplus

/* OMTensor creator with data sizes and element type  */
template <typename T>
OMTensor *omTensorCreateWithShape(vector<int64_t> dataSizes) {
  /* Create a OMTensor with data sizes and strides allocated */
  auto omt = omTensorCreateEmpty(dataSizes.size());
  if (omt == NULL)
    return NULL;

  /* Allocate data buffer */
  omt->_rank = dataSizes.size();
  if ((omt->_data = malloc(
           getNumOfElems(dataSizes.data(), omt->_rank) * sizeof(T))) == NULL) {
    omTensorDestroy(omt);
    return NULL;
  }

  omt->_alignedData = omt->_data;
  omt->_offset = 0;

  /* Copy dataSizes, _dataSizes already allocated by omTensorCreate */
  copy(dataSizes.begin(), dataSizes.end(), omt->_dataSizes);

  /* Compute and copy dataStrides, _dataStrides already allocated by omTensorCreateEmpty
   */
  auto computedStrides = computeStridesFromSizes(omt->_dataSizes, omt->_rank);
  copy(computedStrides.begin(), computedStrides.end(), omt->_dataStrides);

  /* Convert CPP type to ONNX type */
  try {
    omt->_dataType =
        OM_DATA_TYPE_CPP_TO_ONNX.at(string(typeid(T).name()));
  } catch (const out_of_range &e) {
    omt->_dataType = ONNX_TYPE_UNDEFINED;
  }

  /* Set flag for destructor */
  omt->_owningData = true;

  return omt;
}

/* OMTensor creator with data sizes, element type and random data */
template <typename T>
OMTensor *omTensorCreateWithRandomData(
    vector<int64_t> dataSizes, T lbound, T ubound) {
  // Will be used to obtain a seed for the random number engine
  random_device rd;
  // Standard mersenne_twister_engine seeded with rd()
  mt19937 gen(rd());
  uniform_real_distribution<> dis(lbound, ubound);

  auto omt = omTensorCreateWithShape<T>(dataSizes);
  if (omt == NULL)
    return NULL;

  generate((T *)omt->_data,
      (T *)omt->_data + getNumOfElems(omt->_dataSizes, omt->_rank),
      [&]() { return dis(gen); });
  return omt;
}

/* OMTensor aligned data getter */
void *omTensorGetAlignedData(OMTensor *omt) { return omt->_alignedData; }

/* OMTensor aligned data setter */
void omTensorSetAlignedData(OMTensor *omt, void *alignedData) {
  omt->_alignedData = alignedData;
}

/* Access an element (by reference) at offset computed by index array */
template <typename T>
T &omTensorGetElem(OMTensor *omt, std::vector<int64_t> indexes) {
  int64_t elemOffset = omTensorComputeElemOffset(omt, indexes);
  return ((T *)omt->_data)[elemOffset];
}

/* Access an element (by reference) at linear offset */
template <typename T>
T &omTensorGetElemByOffset(OMTensor *omt, int64_t index) {
  return ((T *)omt->_data)[index];
}

/* Compute strides vector from sizes vector */
vector<int64_t> omTensorComputeStridesFromShape(OMTensor *omt) {
  return computeStridesFromSizes(omt->_dataSizes, omt->_rank);
}

/* Compute linear element offset from multi-dimensional index array */
int64_t omTensorComputeElemOffset(OMTensor *omt, vector<int64_t> &indexes) {
  return computeElemOffset(omt->_dataStrides, omt->_rank, indexes);
}

/* Compute index set for the whole OMTensor */
vector<vector<int64_t>> omTensorComputeIndexSet(OMTensor *omt) {
  // First, we create index set of each dimension separately.
  // i.e., for a tensor/OMT of shape (2, 3), its dimWiseIdxSet will be:
  // {{0,1}, {0,1,2}};
  vector<vector<int64_t>> dimWiseIdxSet;
  for (auto dimSize :
      vector<int64_t>(omt->_dataSizes, omt->_dataSizes + omt->_rank)) {
    vector<int64_t> dimIdxSet(dimSize);
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
  auto aShape = vector<int64_t>(a->_dataSizes, a->_dataSizes + a->_rank);
  auto bShape = vector<int64_t>(b->_dataSizes, b->_dataSizes + b->_rank);
  if (aShape != bShape) {
    cerr << "Shape mismatch ";
    printVector(aShape, ",", cerr);
    cerr << " != ";
    printVector(bShape, ",", cerr);
    return false;
  }

  // Compute absolute difference, verify it's within tolerable range.
  auto anum = omTensorGetNumElems(a);
  vector<T> absoluteDiff(anum);
  transform((T *)a->_data, (T *)a->_data + anum, (T *)b->_data,
      absoluteDiff.begin(), minus<>());
  transform(absoluteDiff.begin(), absoluteDiff.end(), absoluteDiff.begin(),
      static_cast<T (*)(T)>(&abs));
  bool atolSatisfied = all_of(
      absoluteDiff.begin(), absoluteDiff.end(), [&](T a) { return a < atol; });

  // Compute relative difference, verify it's within tolerable range.
  vector<T> relativeDiff(anum);
  transform(absoluteDiff.begin(), absoluteDiff.end(), (T *)a->_data,
      relativeDiff.begin(), divides<>());
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
        cerr << "a[";
        printVector(idx, ",", cerr);
        cerr << "] = " << aElem << " != ";
        cerr << "b[";
        printVector(idx, ",", cerr);
        cerr << "] = " << bElem << endl;
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
template OMTensor *omTensorCreateWithShape<float>(std::vector<int64_t> dataSizes);
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
template float &omTensorGetElem<float>(OMTensor *, std::vector<int64_t> indexes);
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
