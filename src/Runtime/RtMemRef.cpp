//===------------ RtMemRef.cpp - RtMemRef Implementation -------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of RtMemRef data structures
// and helper functions.
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>

#include <malloc.h>

// clang-format off
#include "_RtMemRef.h"
#include "DataType.h"
#include "RtMemRef.hpp"
// clang-format on

using namespace std;

/* ================ External C/C++ API call implementation ================ */

/*----------------------------- */
/* C/C++ API for RtMemRef calls */
/*----------------------------- */

/* RtMemRef creator */
RtMemRef *rmr_create(int rank) {
  try {
    return new RtMemRef(rank);
  } catch (const runtime_error &e) {
    return NULL;
  }
}

/* RtMemRef destroyer */
void rmr_destroy(RtMemRef *rmr) { delete rmr; }

/* RtMemRef data getter */
void *rmr_getData(RtMemRef *rmr) { return rmr->_data; }

/* RtMemRef data setter */
void rmr_setData(RtMemRef *rmr, void *data) {
  /* If we allocated the data buffer, free it first.
   * Once this is done, caller will be responsible for
   * managing the data buffer.
   */
  if (rmr->_dataMalloc) {
    free(rmr->_data);
    rmr->_dataMalloc = false;
  }
  rmr->_data = data;
}

/* RtMemRef data sizes getter */
INDEX_TYPE *rmr_getDataSizes(RtMemRef *rmr) { return rmr->_dataSizes; }

/* RtMemRef data sizes setter */
void rmr_setDataSizes(RtMemRef *rmr, INDEX_TYPE *dataSizes) {
  for (int i = 0; i < rmr->_rank; i++)
    rmr->_dataSizes[i] = dataSizes[i];
}

/* RtMemRef data strides getter */
int64_t *rmr_getDataStrides(RtMemRef *rmr) { return rmr->_dataStrides; }

/* RtMemRef data strides setter */
void rmr_setDataStrides(RtMemRef *rmr, int64_t *dataStrides) {
  for (int i = 0; i < rmr->_rank; i++)
    rmr->_dataStrides[i] = dataStrides[i];
}

/* RtMemRef data type getter */
int rmr_getDataType(RtMemRef *rmr) { return rmr->_dataType; }

/* RtMemRef data type setter */
void rmr_setDataType(RtMemRef *rmr, int dataType) {
  rmr->_dataType =
      dataType < 0 || dataType >= sizeof(RTMEMREF_DATA_TYPE_SIZE) / sizeof(int)
          ? ONNX_TYPE_UNDEFINED
          : dataType;
}

/* RtMemRef data buffer size getter */
int64_t rmr_getDataBufferSize(RtMemRef *rmr) {
  return getNumOfElems(rmr->_dataSizes, rmr->_rank) *
         getDataTypeSize(rmr->_dataType);
}

/* RtMemRef rank getter */
int rmr_getRank(RtMemRef *rmr) { return rmr->_rank; }

/* RtMemRef name getter */
char *rmr_getName(RtMemRef *rmr) { return (char *)rmr->_name.c_str(); }

/* RtMemRef name setter */
void rmr_setName(RtMemRef *rmr, char *name) {
  rmr->_name = name ? string(name) : "";
}

/* RtMemRef number of elements getter */
INDEX_TYPE rmr_getNumOfElems(RtMemRef *rmr) {
  return getNumOfElems(rmr->_dataSizes, rmr->_rank);
}

/*---------------------------------------- */
/* C/C++ API for OrderedRtMemRefDict calls */
/*---------------------------------------- */

/* OrderedRtMemRefDict creator */
OrderedRtMemRefDict *ormrd_create(RtMemRef *rmrs[], int n) {
  try {
    return new OrderedRtMemRefDict(rmrs, n);
  } catch (const invalid_argument &e) {
    return NULL;
  }
}

/* OrderedRtMemRefDict destroyer */
void ormrd_destroy(OrderedRtMemRefDict *ormrd) { delete ormrd; }

/* OrderedRtMemRefDict RtMemRef array getter */
RtMemRef **ormrd_getRmrs(OrderedRtMemRefDict *ormrd) {
  return ormrd->_rmrs.data();
}

/* OrderedRtMemRefDict number of RtMemRef getter */
int ormrd_getNumOfRmrs(OrderedRtMemRefDict *ormrd) {
  return ormrd->_rmrs.size();
}

/* ================ Internal C++ API call implementation ================ */

#ifdef RTMEMREF_INTERNAL_API

/*----------------------------------------- */
/* C++ API for internal only RtMemRef calls */
/*----------------------------------------- */

/* RtMemRef creator with data sizes and element type  */
template <typename T>
RtMemRef *rmr_createWithDataSizes(vector<INDEX_TYPE> dataSizes) {
  /* Create a RtMemRef with data sizes and strides allocated */
  auto rmr = rmr_create(dataSizes.size());
  if (rmr == NULL)
    return NULL;

  /* Allocate data buffer */
  rmr->_rank = dataSizes.size();
  if ((rmr->_data = malloc(
           getNumOfElems(dataSizes.data(), rmr->_rank) * sizeof(T))) == NULL) {
    rmr_destroy(rmr);
    return NULL;
  }

  rmr->_alignedData = rmr->_data;
  rmr->_offset = 0;

  /* Copy dataSizes, _dataSizes already allocated by rmr_create */
  copy(dataSizes.begin(), dataSizes.end(), rmr->_dataSizes);

  /* Compute and copy dataStrides, _dataStrides already allocated by rmr_create
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
  rmr->_dataMalloc = true;

  return rmr;
}

/* RtMemRef creator with data sizes, element type and random data */
template <typename T>
RtMemRef *rmr_createWithRandomData(
    vector<INDEX_TYPE> dataSizes, T lbound, T ubound) {
  // Will be used to obtain a seed for the random number engine
  random_device rd;
  // Standard mersenne_twister_engine seeded with rd()
  mt19937 gen(rd());
  uniform_real_distribution<> dis(lbound, ubound);

  auto rmr = rmr_createWithDataSizes<T>(dataSizes);
  if (rmr == NULL)
    return NULL;

  generate((T *)rmr->_data,
      (T *)rmr->_data + getNumOfElems(rmr->_dataSizes, rmr->_rank),
      [&]() { return dis(gen); });
  return rmr;
}

/* RtMemRef aligned data getter */
void *rmr_getAlignedData(RtMemRef *rmr) { return rmr->_alignedData; }

/* RtMemRef aligned data setter */
void rmr_setAlignedData(RtMemRef *rmr, void *alignedData) {
  rmr->_alignedData = alignedData;
}

/* Access an element (by reference) at offset computed by index array */
template <typename T>
T &rmr_getElemByOffset(RtMemRef *rmr, std::vector<INDEX_TYPE> indexes) {
  INDEX_TYPE elemOffset = rmr_computeElemOffset(rmr, indexes);
  return ((T *)rmr->_data)[elemOffset];
}

/* Access an element (by reference) at linear offset */
template <typename T>
T &rmr_getElemByIndex(RtMemRef *rmr, INDEX_TYPE index) {
  return ((T *)rmr->_data)[index];
}

/* Compute strides vector from sizes vector */
vector<int64_t> rmr_computeStridesFromSizes(RtMemRef *rmr) {
  return computeStridesFromSizes(rmr->_dataSizes, rmr->_rank);
}

/* Compute linear element offset from multi-dimensional index array */
INDEX_TYPE rmr_computeElemOffset(RtMemRef *rmr, vector<INDEX_TYPE> &indexes) {
  return computeElemOffset(rmr->_dataStrides, rmr->_rank, indexes);
}

/* Compute index set for the whole RtMemRef */
vector<vector<INDEX_TYPE>> rmr_computeIndexSet(RtMemRef *rmr) {
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

/* Check whether two RtMemRef data are "close" to each other */
template <typename T>
inline bool rmr_areTwoRmrsClose(
    RtMemRef *a, RtMemRef *b, float rtol, float atol) {

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
  auto anum = rmr_getNumOfElems(a);
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
    for (const auto &idx : rmr_computeIndexSet(a)) {
      T aElem = rmr_getElemByOffset<T>(a, idx);
      T bElem = rmr_getElemByOffset<T>(b, idx);
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
/* C++ API for internal only OrderedRtMemRefDict calls */
/*---------------------------------------------------- */

/* Create an empty OrderedRtMemRefDict so RtMemRef can be added one by one. */
OrderedRtMemRefDict *ormrd_create(void) { return new OrderedRtMemRefDict(); }

/* Return RtMemRef at specified index in the OrderedRtMemRefDict */
RtMemRef *ormrd_getRmrByIndex(OrderedRtMemRefDict *ormrd, int index) {
  assert(index >= 0);
  return index < ormrd->_rmrs.size() ? ormrd->_rmrs[index] : NULL;
}

/* Set RtMemRef at specified index in the OrderedRtMemRefDict
 *
 * Currently,
 * - attempting to set RtMemRef at the same index more than once is a bug
 * - attempting to set RtMemRef with a name that already exists is a bug
 */
void ormrd_setRmrByIndex(OrderedRtMemRefDict *ormrd, RtMemRef *rmr, int index) {
  if (index < ormrd->_rmrs.size())
    assert(index >= 0 && ormrd->_rmrs[index] == NULL);
  else
    ormrd->_rmrs.resize(index + 1);

  /* if the RtMemRef has a name, try to create the name to index mapping */
  if (!rmr->_name.empty()) {
    auto ret = ormrd->_n2imap.insert({rmr->_name, index});
    assert(ret.second == true && "duplicate RtMemRef name");
  }
  ormrd->_rmrs[index] = rmr;
}

/* Return RtMemRef of specified name in the OrderedRtMemRefDict */
RtMemRef *ormrd_getRmrByName(OrderedRtMemRefDict *ormrd, string name) {
  return ormrd->_n2imap[name] ? ormrd->_rmrs[ormrd->_n2imap[name]] : NULL;
}

/* Force the compiler to instantiate the template functions and
 * include them in the library
 */
static void __dummy_donot_call__(void) {
  rmr_createWithDataSizes<int32_t>(vector<INDEX_TYPE>{});
  rmr_createWithDataSizes<int64_t>(vector<INDEX_TYPE>{});
  rmr_createWithDataSizes<float>(vector<INDEX_TYPE>{});
  rmr_createWithDataSizes<double>(vector<INDEX_TYPE>{});

  rmr_createWithRandomData<int32_t>(vector<INDEX_TYPE>{});
  rmr_createWithRandomData<int64_t>(vector<INDEX_TYPE>{});
  rmr_createWithRandomData<float>(vector<INDEX_TYPE>{});
  rmr_createWithRandomData<double>(vector<INDEX_TYPE>{});

  rmr_getElemByOffset<int32_t>(NULL, vector<INDEX_TYPE>{});
  rmr_getElemByOffset<int64_t>(NULL, vector<INDEX_TYPE>{});
  rmr_getElemByOffset<float>(NULL, vector<INDEX_TYPE>{});
  rmr_getElemByOffset<double>(NULL, vector<INDEX_TYPE>{});

  rmr_getElemByIndex<int32_t>(NULL, 0);
  rmr_getElemByIndex<int64_t>(NULL, 0);
  rmr_getElemByIndex<float>(NULL, 0);
  rmr_getElemByIndex<double>(NULL, 0);

  rmr_areTwoRmrsClose<int32_t>(NULL, NULL);
  rmr_areTwoRmrsClose<int64_t>(NULL, NULL);
  rmr_areTwoRmrsClose<float>(NULL, NULL);
  rmr_areTwoRmrsClose<double>(NULL, NULL);
}

#endif
