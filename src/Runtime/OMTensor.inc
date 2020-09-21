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
OMTensor *rmrCreate(int rank) {
  OMTensor *rmr = (OMTensor *)malloc(sizeof(struct OMTensor));
  if ((rmr->_dataSizes = (INDEX_TYPE *)malloc(rank * sizeof(INDEX_TYPE))) &&
      (rmr->_dataStrides = (int64_t *)malloc(rank * sizeof(int64_t)))) {
    rmr->_data = NULL;
    rmr->_alignedData = NULL;
    rmr->_offset = 0;
    rmr->_dataType = ONNX_TYPE_UNDEFINED;
    rmr->_rank = rank;
    rmr->_name = NULL;
    rmr->_owningData = false;
  }
  return rmr;
}

/* More detailed OMTensor creator */
OMTensor *rmrCreateWithNameAndOwnership(int rank, char *name, bool ownership) {
  OMTensor *rmr = (OMTensor *)malloc(sizeof(struct OMTensor));
  ;
  if ((rmr->_dataSizes = (INDEX_TYPE *)malloc(rank * sizeof(INDEX_TYPE))) &&
      (rmr->_dataStrides = (int64_t *)malloc(rank * sizeof(int64_t)))) {
    rmr->_data = NULL;
    rmr->_alignedData = NULL;
    rmr->_offset = 0;
    rmr->_dataType = ONNX_TYPE_UNDEFINED;
    rmr->_rank = rank;
    rmr->_name = name;
    rmr->_owningData = ownership;
  }
}

/* OMTensor destroyer */
// TODO(tjingrant): properly clean up with respect to ownership.
void rmrDestroy(OMTensor *rmr) { free(rmr); }

/* OMTensor data getter */
void *rmrGetData(OMTensor *rmr) { return rmr->_data; }

/* OMTensor data setter */
void rmrSetData(OMTensor *rmr, void *data) {
  /* If we own the data buffer, free it first. */
  if (rmr->_owningData) {
    free(rmr->_data);
  }
  rmr->_data = data;
}

/* OMTensor data sizes getter */
INDEX_TYPE *rmrGetDataShape(OMTensor *rmr) { return rmr->_dataSizes; }

/* OMTensor data sizes setter */
void rmrSetDataShape(OMTensor *rmr, INDEX_TYPE *dataSizes) {
  for (int i = 0; i < rmr->_rank; i++)
    rmr->_dataSizes[i] = dataSizes[i];
}

/* OMTensor data strides getter */
int64_t *rmrGetDataStrides(OMTensor *rmr) { return rmr->_dataStrides; }

/* OMTensor data strides setter */
void rmrSetDataStrides(OMTensor *rmr, int64_t *dataStrides) {
  for (int i = 0; i < rmr->_rank; i++)
    rmr->_dataStrides[i] = dataStrides[i];
}

/* OMTensor data type getter */
int rmrGetDataType(OMTensor *rmr) { return rmr->_dataType; }

/* OMTensor data type setter */
void rmrSetDataType(OMTensor *rmr, int dataType) {
  rmr->_dataType =
      dataType < 0 || dataType >= sizeof(RTMEMREF_DATA_TYPE_SIZE) / sizeof(int)
          ? ONNX_TYPE_UNDEFINED
          : dataType;
}

/* OMTensor data buffer size getter */
int64_t rmrGetDataBufferSize(OMTensor *rmr) {
  return getNumOfElems(rmr->_dataSizes, rmr->_rank) *
         getDataTypeSize(rmr->_dataType);
}

/* OMTensor rank getter */
int rmrGetRank(OMTensor *rmr) { return rmr->_rank; }

/* OMTensor name getter */
char *rmrGetName(OMTensor *rmr) { return rmr->_name; }

/* OMTensor name setter */
void rmrSetName(OMTensor *rmr, char *name) { rmr->_name = name; }

/* OMTensor number of elements getter */
INDEX_TYPE rmrGetNumElems(OMTensor *rmr) {
  return getNumOfElems(rmr->_dataSizes, rmr->_rank);
}

/*---------------------------------------- */
/* C/C++ API for OMTensorList calls */
/*---------------------------------------- */

/* OMTensorList creator */
OMTensorList *rmrListCreate(OMTensor **rmrs, int n) {
  OMTensorList *rmrList = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  rmrList->_rmrs = rmrs;
  rmrList->_n = n;
  return rmrList;
  //  try {
  //    return new OMTensorList(rmrs, n);
  //  } catch (const invalid_argument &e) {
  //    return NULL;
  //  }
}

/* OMTensorList destroyer */
void rmrListDestroy(OMTensorList *rmrList) {
  for (int i = 0; i < rmrList->_n; i++)
    free(rmrList->_rmrs[i]);
  free(rmrList);
}

/* OMTensorList OMTensor array getter */
OMTensor **rmrListGetPtrToRmrs(OMTensorList *ormrd) { return ormrd->_rmrs; }

/* OMTensorList number of OMTensor getter */
int rmrListGetNumRmrs(OMTensorList *ormrd) { return ormrd->_n; }

/* Return OMTensor at specified index in the OMTensorList */
OMTensor *rmrListGetRmrByIndex(OMTensorList *rlist, int index) {
  assert(index >= 0);
  assert(index < rlist->_n);
  return rlist->_rmrs[index];
}

/* ================ C++ API call implementation ================ */
#ifdef __cplusplus

/* OMTensor creator with data sizes and element type  */
template <typename T>
OMTensor *rmrCreateWithShape(vector<INDEX_TYPE> dataSizes) {
  /* Create a OMTensor with data sizes and strides allocated */
  auto rmr = rmrCreate(dataSizes.size());
  if (rmr == NULL)
    return NULL;

  /* Allocate data buffer */
  rmr->_rank = dataSizes.size();
  if ((rmr->_data = malloc(
           getNumOfElems(dataSizes.data(), rmr->_rank) * sizeof(T))) == NULL) {
    rmrDestroy(rmr);
    return NULL;
  }

  rmr->_alignedData = rmr->_data;
  rmr->_offset = 0;

  /* Copy dataSizes, _dataSizes already allocated by rmrCreate */
  copy(dataSizes.begin(), dataSizes.end(), rmr->_dataSizes);

  /* Compute and copy dataStrides, _dataStrides already allocated by rmrCreate
   */
  auto computedStrides = computeStridesFromSizes(rmr->_dataSizes, rmr->_rank);
  copy(computedStrides.begin(), computedStrides.end(), rmr->_dataStrides);

  /* Convert CPP type to ONNX type */
  try {
    rmr->_dataType =
        RTMEMREF_DATA_TYPE_CPP_TO_ONNX.at(string(typeid(T).name()));
  } catch (const out_of_range &e) {
    rmr->_dataType = ONNX_TYPE_UNDEFINED;
  }

  /* Set flag for destructor */
  rmr->_owningData = true;

  return rmr;
}

/* OMTensor creator with data sizes, element type and random data */
template <typename T>
OMTensor *rmrCreateWithRandomData(
    vector<INDEX_TYPE> dataSizes, T lbound, T ubound) {
  // Will be used to obtain a seed for the random number engine
  random_device rd;
  // Standard mersenne_twister_engine seeded with rd()
  mt19937 gen(rd());
  uniform_real_distribution<> dis(lbound, ubound);

  auto rmr = rmrCreateWithShape<T>(dataSizes);
  if (rmr == NULL)
    return NULL;

  generate((T *)rmr->_data,
      (T *)rmr->_data + getNumOfElems(rmr->_dataSizes, rmr->_rank),
      [&]() { return dis(gen); });
  return rmr;
}

/* OMTensor aligned data getter */
void *rmrGetAlignedData(OMTensor *rmr) { return rmr->_alignedData; }

/* OMTensor aligned data setter */
void rmrSetAlignedData(OMTensor *rmr, void *alignedData) {
  rmr->_alignedData = alignedData;
}

/* Access an element (by reference) at offset computed by index array */
template <typename T>
T &rmrGetElem(OMTensor *rmr, std::vector<INDEX_TYPE> indexes) {
  INDEX_TYPE elemOffset = rmrComputeElemOffset(rmr, indexes);
  return ((T *)rmr->_data)[elemOffset];
}

/* Access an element (by reference) at linear offset */
template <typename T>
T &rmrGetElemByOffset(OMTensor *rmr, INDEX_TYPE index) {
  return ((T *)rmr->_data)[index];
}

/* Compute strides vector from sizes vector */
vector<int64_t> rmrComputeStridesFromShape(OMTensor *rmr) {
  return computeStridesFromSizes(rmr->_dataSizes, rmr->_rank);
}

/* Compute linear element offset from multi-dimensional index array */
INDEX_TYPE rmrComputeElemOffset(OMTensor *rmr, vector<INDEX_TYPE> &indexes) {
  return computeElemOffset(rmr->_dataStrides, rmr->_rank, indexes);
}

/* Compute index set for the whole OMTensor */
vector<vector<INDEX_TYPE>> rmrComputeIndexSet(OMTensor *rmr) {
  // First, we create index set of each dimension separately.
  // i.e., for a tensor/RMR of shape (2, 3), its dimWiseIdxSet will be:
  // {{0,1}, {0,1,2}};
  vector<vector<INDEX_TYPE>> dimWiseIdxSet;
  for (auto dimSize :
      vector<INDEX_TYPE>(rmr->_dataSizes, rmr->_dataSizes + rmr->_rank)) {
    vector<INDEX_TYPE> dimIdxSet(dimSize);
    iota(begin(dimIdxSet), end(dimIdxSet), 0);
    dimWiseIdxSet.emplace_back(dimIdxSet);
  }
  // Then, the cartesian product of vectors within dimWiseIdxSet will be the
  // index set for the whole RMR.
  return CartProduct(dimWiseIdxSet);
}

/* Check whether two OMTensor data are "close" to each other */
template <typename T>
inline bool rmrAreTwoRmrsClose(
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
  auto anum = rmrGetNumElems(a);
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
    for (const auto &idx : rmrComputeIndexSet(a)) {
      T aElem = rmrGetElem<T>(a, idx);
      T bElem = rmrGetElem<T>(b, idx);
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
OMTensor *rmrListGetRmrByName(OMTensorList *rlist, string name) {
  for (int i = 0; i < rlist->_n; i++)
    if (rlist->_rmrs[i]->_name == name)
      return rlist->_rmrs[i];
  return NULL;
}

// Explicit instantiation of all templated API functions.

template OMTensor *rmrCreateWithShape<int32_t>(
    std::vector<INDEX_TYPE> dataSizes);
template OMTensor *rmrCreateWithShape<int64_t>(
    std::vector<INDEX_TYPE> dataSizes);
template OMTensor *rmrCreateWithShape<float>(std::vector<INDEX_TYPE> dataSizes);
template OMTensor *rmrCreateWithShape<double>(
    std::vector<INDEX_TYPE> dataSizes);

template OMTensor *rmrCreateWithRandomData<int32_t>(
    std::vector<INDEX_TYPE> dataSizes, int32_t lbound, int32_t ubound);
template OMTensor *rmrCreateWithRandomData<int64_t>(
    std::vector<INDEX_TYPE> dataSizes, int64_t lbound, int64_t ubound);
template OMTensor *rmrCreateWithRandomData<float>(
    std::vector<INDEX_TYPE> dataSizes, float lbound, float ubound);
template OMTensor *rmrCreateWithRandomData<double>(
    std::vector<INDEX_TYPE> dataSizes, double lbound, double ubound);

template int32_t &rmrGetElem<int32_t>(
    OMTensor *, std::vector<INDEX_TYPE> indexes);
template int64_t &rmrGetElem<int64_t>(
    OMTensor *, std::vector<INDEX_TYPE> indexes);
template float &rmrGetElem<float>(OMTensor *, std::vector<INDEX_TYPE> indexes);
template double &rmrGetElem<double>(
    OMTensor *, std::vector<INDEX_TYPE> indexes);

template int32_t &rmrGetElemByOffset<int32_t>(OMTensor *, INDEX_TYPE index);
template int64_t &rmrGetElemByOffset<int64_t>(OMTensor *, INDEX_TYPE index);
template float &rmrGetElemByOffset<float>(OMTensor *, INDEX_TYPE indexs);
template double &rmrGetElemByOffset<double>(OMTensor *, INDEX_TYPE index);

template bool rmrAreTwoRmrsClose<int32_t>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
template bool rmrAreTwoRmrsClose<int64_t>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
template bool rmrAreTwoRmrsClose<float>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
template bool rmrAreTwoRmrsClose<double>(
    OMTensor *a, OMTensor *b, float rtol, float atol);
#endif
