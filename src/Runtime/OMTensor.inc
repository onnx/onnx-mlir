//===------------ OMTensor.cpp - OMTensor Implementation -------------===//
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

// clang-format off
#include "OnnxMlirInternal.h"
#include "OMTensor.h"
// clang-format on

#ifdef __cplusplus
using namespace std;
#endif

/* ================ External C/C++ API call implementation ================ */

/*----------------------------- */
/* C/C++ API for OMTensor calls */
/*----------------------------- */

/* OMTensor creator */
OMTensor *omtCreate(int rank) {
  OMTensor *omt = (OMTensor *)malloc(sizeof(struct OMTensor));
  if ((omt->_dataSizes = (INDEX_TYPE *)malloc(rank * sizeof(INDEX_TYPE))) &&
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
OMTensor *omtCreateWithNameAndOwnership(int rank, char *name, bool ownership) {
  OMTensor *omt = (OMTensor *)malloc(sizeof(struct OMTensor));
  ;
  if ((omt->_dataSizes = (INDEX_TYPE *)malloc(rank * sizeof(INDEX_TYPE))) &&
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
void omtDestroy(OMTensor *omt) { free(omt); }

/* OMTensor data getter */
void *omtGetData(OMTensor *omt) { return omt->_data; }

/* OMTensor data setter */
void omtSetData(OMTensor *omt, void *data) {
  /* If we own the data buffer, free it first. */
  if (omt->_owningData) {
    free(omt->_data);
  }
  omt->_data = data;
}

/* OMTensor data sizes getter */
INDEX_TYPE *omtGetDataShape(OMTensor *omt) { return omt->_dataSizes; }

/* OMTensor data sizes setter */
void omtSetDataShape(OMTensor *omt, INDEX_TYPE *dataSizes) {
  for (int i = 0; i < omt->_rank; i++)
    omt->_dataSizes[i] = dataSizes[i];
}

/* OMTensor data strides getter */
int64_t *omtGetDataStrides(OMTensor *omt) { return omt->_dataStrides; }

/* OMTensor data strides setter */
void omtSetDataStrides(OMTensor *omt, int64_t *dataStrides) {
  for (int i = 0; i < omt->_rank; i++)
    omt->_dataStrides[i] = dataStrides[i];
}

/* OMTensor data type getter */
int omtGetDataType(OMTensor *omt) { return omt->_dataType; }

/* OMTensor data type setter */
void omtSetDataType(OMTensor *omt, int dataType) {
  omt->_dataType =
      dataType < 0 || dataType >= sizeof(RTMEMREF_DATA_TYPE_SIZE) / sizeof(int)
          ? ONNX_TYPE_UNDEFINED
          : dataType;
}

/* OMTensor data buffer size getter */
int64_t omtGetDataBufferSize(OMTensor *omt) {
  return getNumOfElems(omt->_dataSizes, omt->_rank) *
         getDataTypeSize(omt->_dataType);
}

/* OMTensor rank getter */
int omtGetRank(OMTensor *omt) { return omt->_rank; }

/* OMTensor name getter */
char *omtGetName(OMTensor *omt) { return omt->_name; }

/* OMTensor name setter */
void omtSetName(OMTensor *omt, char *name) { omt->_name = name; }

/* OMTensor number of elements getter */
INDEX_TYPE omtGetNumElems(OMTensor *omt) {
  return getNumOfElems(omt->_dataSizes, omt->_rank);
}

/*---------------------------------------- */
/* C/C++ API for OMTensorList calls */
/*---------------------------------------- */

/* OMTensorList creator */
OMTensorList *omtListCreate(OMTensor **omts, int n) {
  OMTensorList *omtList = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  omtList->_omts = omts;
  omtList->_n = n;
  return omtList;
  //  try {
  //    return new OMTensorList(omts, n);
  //  } catch (const invalid_argument &e) {
  //    return NULL;
  //  }
}

/* OMTensorList destroyer */
void omtListDestroy(OMTensorList *omtList) {
  for (int i = 0; i < omtList->_n; i++)
    free(omtList->_omts[i]);
  free(omtList);
}

/* OMTensorList OMTensor array getter */
OMTensor **omtListGetPtrToOmts(OMTensorList *list) { return list->_omts; }

/* OMTensorList number of OMTensor getter */
int omtListGetNumOmts(OMTensorList *list) { return list->_n; }

/* Return OMTensor at specified index in the OMTensorList */
OMTensor *omtListGetOmtByIndex(OMTensorList *rlist, int index) {
  assert(index >= 0);
  assert(index < rlist->_n);
  return rlist->_omts[index];
}

/* ================ C++ API call implementation ================ */
#ifdef __cplusplus

/* OMTensor creator with data sizes and element type  */
template <typename T>
OMTensor *omtCreateWithShape(vector<INDEX_TYPE> dataSizes) {
  /* Create a OMTensor with data sizes and strides allocated */
  auto omt = omtCreate(dataSizes.size());
  if (omt == NULL)
    return NULL;

  /* Allocate data buffer */
  omt->_rank = dataSizes.size();
  if ((omt->_data = malloc(
           getNumOfElems(dataSizes.data(), omt->_rank) * sizeof(T))) == NULL) {
    omtDestroy(omt);
    return NULL;
  }

  omt->_alignedData = omt->_data;
  omt->_offset = 0;

  /* Copy dataSizes, _dataSizes already allocated by omtCreate */
  copy(dataSizes.begin(), dataSizes.end(), omt->_dataSizes);

  /* Compute and copy dataStrides, _dataStrides already allocated by omtCreate
   */
  auto computedStrides = computeStridesFromSizes(omt->_dataSizes, omt->_rank);
  copy(computedStrides.begin(), computedStrides.end(), omt->_dataStrides);

  /* Convert CPP type to ONNX type */
  try {
    omt->_dataType =
        RTMEMREF_DATA_TYPE_CPP_TO_ONNX.at(string(typeid(T).name()));
  } catch (const out_of_range &e) {
    omt->_dataType = ONNX_TYPE_UNDEFINED;
  }

  /* Set flag for destructor */
  omt->_owningData = true;

  return omt;
}

/* OMTensor creator with data sizes, element type and random data */
template <typename T>
OMTensor *omtCreateWithRandomData(
    vector<INDEX_TYPE> dataSizes, T lbound, T ubound) {
  // Will be used to obtain a seed for the random number engine
  random_device rd;
  // Standard mersenne_twister_engine seeded with rd()
  mt19937 gen(rd());
  uniform_real_distribution<> dis(lbound, ubound);

  auto omt = omtCreateWithShape<T>(dataSizes);
  if (omt == NULL)
    return NULL;

  generate((T *)omt->_data,
      (T *)omt->_data + getNumOfElems(omt->_dataSizes, omt->_rank),
      [&]() { return dis(gen); });
  return omt;
}

/* OMTensor aligned data getter */
void *omtGetAlignedData(OMTensor *omt) { return omt->_alignedData; }

/* OMTensor aligned data setter */
void omtSetAlignedData(OMTensor *omt, void *alignedData) {
  omt->_alignedData = alignedData;
}

/* Access an element (by reference) at offset computed by index array */
template <typename T>
T &omtGetElem(OMTensor *omt, std::vector<INDEX_TYPE> indexes) {
  INDEX_TYPE elemOffset = omtComputeElemOffset(omt, indexes);
  return ((T *)omt->_data)[elemOffset];
}

/* Access an element (by reference) at linear offset */
template <typename T>
T &omtGetElemByOffset(OMTensor *omt, INDEX_TYPE index) {
  return ((T *)omt->_data)[index];
}

/* Compute strides vector from sizes vector */
vector<int64_t> omtComputeStridesFromShape(OMTensor *omt) {
  return computeStridesFromSizes(omt->_dataSizes, omt->_rank);
}

/* Compute linear element offset from multi-dimensional index array */
INDEX_TYPE omtComputeElemOffset(OMTensor *omt, vector<INDEX_TYPE> &indexes) {
  return computeElemOffset(omt->_dataStrides, omt->_rank, indexes);
}

/* Compute index set for the whole OMTensor */
vector<vector<INDEX_TYPE>> omtComputeIndexSet(OMTensor *omt) {
  // First, we create index set of each dimension separately.
  // i.e., for a tensor/OMT of shape (2, 3), its dimWiseIdxSet will be:
  // {{0,1}, {0,1,2}};
  vector<vector<INDEX_TYPE>> dimWiseIdxSet;
  for (auto dimSize :
      vector<INDEX_TYPE>(omt->_dataSizes, omt->_dataSizes + omt->_rank)) {
    vector<INDEX_TYPE> dimIdxSet(dimSize);
    iota(begin(dimIdxSet), end(dimIdxSet), 0);
    dimWiseIdxSet.emplace_back(dimIdxSet);
  }
  // Then, the cartesian product of vectors within dimWiseIdxSet will be the
  // index set for the whole OMT.
  return CartProduct(dimWiseIdxSet);
}

/* Check whether two OMTensor data are "close" to each other */
template <typename T>
inline bool omtAreTwoOmtsClose(
    OMTensor *a, OMTensor *b, float rtol, float atol) {

  // Compare shape.
  auto aShape = vector<INDEX_TYPE>(a->_dataSizes, a->_dataSizes + a->_rank);
  auto bShape = vector<INDEX_TYPE>(b->_dataSizes, b->_dataSizes + b->_rank);
  if (aShape != bShape) {
    cerr << "Shape mismatch ";
    printVector(aShape, ",", cerr);
    cerr << " != ";
    printVector(bShape, ",", cerr);
    return false;
  }

  // Compute absolute difference, verify it's within tolerable range.
  auto anum = omtGetNumElems(a);
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
    for (const auto &idx : omtComputeIndexSet(a)) {
      T aElem = omtGetElem<T>(a, idx);
      T bElem = omtGetElem<T>(b, idx);
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

/*---------------------------------------------------- */
/* C++ API for internal only OMTensorList calls */
/*---------------------------------------------------- */

/* Return OMTensor of specified name in the OMTensorList */
OMTensor *omtListGetOmtByName(OMTensorList *rlist, string name) {
  for (int i = 0; i < rlist->_n; i++)
    if (rlist->_omts[i]->_name == name)
      return rlist->_omts[i];
  return NULL;
}

// Explicit instantiation of all templated API functions.

template OMTensor *omtCreateWithShape<int32_t>(
    std::vector<INDEX_TYPE> dataSizes);
template OMTensor *omtCreateWithShape<int64_t>(
    std::vector<INDEX_TYPE> dataSizes);
template OMTensor *omtCreateWithShape<float>(std::vector<INDEX_TYPE> dataSizes);
template OMTensor *omtCreateWithShape<double>(
    std::vector<INDEX_TYPE> dataSizes);

template OMTensor *omtCreateWithRandomData<int32_t>(
    std::vector<INDEX_TYPE> dataSizes, int32_t lbound, int32_t ubound);
template OMTensor *omtCreateWithRandomData<int64_t>(
    std::vector<INDEX_TYPE> dataSizes, int64_t lbound, int64_t ubound);
template OMTensor *omtCreateWithRandomData<float>(
    std::vector<INDEX_TYPE> dataSizes, float lbound, float ubound);
template OMTensor *omtCreateWithRandomData<double>(
    std::vector<INDEX_TYPE> dataSizes, double lbound, double ubound);

template int32_t &omtGetElem<int32_t>(
    OMTensor *, std::vector<INDEX_TYPE> indexes);
template int64_t &omtGetElem<int64_t>(
    OMTensor *, std::vector<INDEX_TYPE> indexes);
template float &omtGetElem<float>(OMTensor *, std::vector<INDEX_TYPE> indexes);
template double &omtGetElem<double>(
    OMTensor *, std::vector<INDEX_TYPE> indexes);

template int32_t &omtGetElemByOffset<int32_t>(OMTensor *, INDEX_TYPE index);
template int64_t &omtGetElemByOffset<int64_t>(OMTensor *, INDEX_TYPE index);
template float &omtGetElemByOffset<float>(OMTensor *, INDEX_TYPE indexs);
template double &omtGetElemByOffset<double>(OMTensor *, INDEX_TYPE index);

template bool omtAreTwoOmtsClose<int32_t>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
template bool omtAreTwoOmtsClose<int64_t>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
template bool omtAreTwoOmtsClose<float>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
template bool omtAreTwoOmtsClose<double>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
#endif
