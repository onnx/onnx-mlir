/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OMTensorList.cpp - OMTensor C/C++ Implementation ----------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ neutral implementation of OMTensorList data
// structures and helper functions.
//
//===----------------------------------------------------------------------===//

#if defined(__APPLE__) || defined(__MVS__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#include "onnx-mlir/Runtime/OMTensorList.h"
#include <string.h>

#define OM_DRIVER_TIMING 1
#include "src/Runtime/OMInstrumentHelper.h"

struct OMTensorList {
#ifdef __cplusplus

  /**
   * Constructor
   *
   * Create an OMTensorList with specified OMTensor pointer array, the size of
   * the array and the array and indication of whether the destructor should
   * use OMTensorListDestroyShallow
   */
  OMTensorList(OMTensor *omts[], int64_t n, bool shallow = false)
      : _omts(omts), _size(n), _shallow(shallow){};

  /**
   * Destructor
   *
   * Destroy the OMTensorList struct.
   */
  ~OMTensorList() {
    /* Destroy all the OMTensors */
    if (!_shallow)
      for (int64_t i = 0; i < _size; i++)
        omTensorDestroy(_omts[i]);
    free(_omts);
  };
#endif

  /* To facilitate user facing API getOmts, OMTensors are kept in a vector
   * that can be quickly returned as an array. A name to index map is used
   * to address ReMemRefs by name.
   */
  OMTensor **_omts; // OMTensor array

  int64_t _size; // Number of elements in _omts.

  bool _shallow; // Internal boolean for the C++ constructor/destructor API
                 // This indicates whether to perform a shallow destroy.
};

/* OMTensorList creator */
OMTensorList *omTensorListCreate(OMTensor **tensors, int64_t n) {
  OMTensorList *list = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  if (!list)
    return NULL;

  TIMING_INIT_START(tensor_list_create)
  size_t omts_bytes = sizeof(OMTensor *) * n;
  list->_size = n;
  list->_omts = (OMTensor **)malloc(omts_bytes);

  if (!list->_omts) {
    free(list); // free the previously allocated memory in case of an error
    return NULL;
  }
  // Copy the given OMTensors pointers to an array owned by OMTensorList
  memcpy(list->_omts, tensors, omts_bytes);
  TIMING_STOP_PRINT(tensor_list_create);
  return list;
}

/* OMTensorList destroyer */
void omTensorListDestroy(OMTensorList *list) {
  if (!list)
    return;
  TIMING_INIT_START(tensor_list_destroy)
  for (int64_t i = 0; i < list->_size; i++)
    omTensorDestroy(list->_omts[i]);
  // Free the list as well as the pointers to the OMTensor array
  omTensorListDestroyShallow(list);
  TIMING_STOP_PRINT(tensor_list_destroy)
}

/* OMTensorList destroyer which does not destroy the tensors.
 */
void omTensorListDestroyShallow(OMTensorList *list) {
  if (!list)
    return;
  // Free the list as well as the pointer to the OMTensor array
  free(list->_omts);
  free(list);
}

/* OMTensorList OMTensor array getter */
OMTensor **omTensorListGetOmtArray(const OMTensorList *list) { return list->_omts; }

/* OMTensorList number of OMTensor getter */
int64_t omTensorListGetSize(const OMTensorList *list) { return list->_size; }

/* Return OMTensor at specified index in the OMTensorList */
OMTensor *omTensorListGetOmtByIndex(const OMTensorList *rlist, int64_t index) {
  assert(index >= 0);
  assert(index < rlist->_size);
  return rlist->_omts[index];
}
