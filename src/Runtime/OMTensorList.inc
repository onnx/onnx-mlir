/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OMTensorList.cpp - OMTensor C/C++ Implementation ----------===//
//
// Copyright 2019-2020 The IBM Research Authors.
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

struct OMTensorList {
#ifdef __cplusplus
  /**
   * Constructor
   *
   * Create an OMTensorList with specified OMTensor pointer array
   * and the size of the array
   */
  OMTensorList(OMTensor *omts[], int64_t n) : _omts(omts), _size(n) {
    _owning = false;
  };

  /**
   * Constructor
   *
   * Create an empty OMTensorList for internal API calls.
   */
  OMTensorList() = default;

  /**
   * Destructor
   *
   * Destroy the OMTensorList struct.
   */
  ~OMTensorList() {
    /* Destroy all the OMTensors */
    for (int64_t i = 0; i < _size; i++)
      omTensorDestroy(_omts[i]);
    if (_owning)
      free(_omts);
  };
#endif

  /* To facilitate user facing API getOmts, OMTensors are kept in a vector
   * that can be quickly returned as an array. A name to index map is used
   * to address ReMemRefs by name.
   */
  OMTensor **_omts; // OMTensor array

  int64_t _size; // Number of elements in _omts.

  int64_t _owning; // indicates whether the OMTensorList owns the pointer to
                   // OMTensor array or not. OMTensorList struct will release
                   // the memory space referred to by '_omts' upon destruction
                   // if and only if it owns it.
};

/* OMTensorList creator */
OMTensorList *omTensorListCreate(OMTensor **tensors, int64_t n) {
  OMTensorList *list = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  if (!list)
    return NULL;
  list->_omts = tensors;
  list->_size = n;
  list->_owning = false;
  return list;
}

/* OMTensorList creator with ownership */
OMTensorList *omTensorListCreateWithOwnership(
    OMTensor **tensors, int64_t n, int64_t owning) {
  OMTensorList *list = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  if (!list)
    return NULL;
  list->_omts = tensors;
  list->_size = n;
  list->_owning = owning;
  return list;
}

/* OMTensorList destroyer */
void omTensorListDestroy(OMTensorList *list) {
  if (!list)
    return;
  for (int64_t i = 0; i < list->_size; i++)
    omTensorDestroy(list->_omts[i]);
  if (list->_owning) {
    free(list->_omts);
  }
  free(list);
}

/* OMTensorList destroyer which does not destroy the tensors. Currently an
 * unpublished call that is mainly used to handle the unique tensor scheme used
 * in Execution Session.
 */
void omTensorListDestroyShallow(OMTensorList *list) {
  if (!list)
    return;
  // Omit destruction of the OMTensors.
  if (list->_owning) {
    free(list->_omts);
  }
  free(list);
}

/* OMTensorList OMTensor array getter */
OMTensor **omTensorListGetOmtArray(OMTensorList *list) { return list->_omts; }

/* OMTensorList number of OMTensor getter */
int64_t omTensorListGetSize(OMTensorList *list) { return list->_size; }

/* Return OMTensor at specified index in the OMTensorList */
OMTensor *omTensorListGetOmtByIndex(OMTensorList *rlist, int64_t index) {
  assert(index >= 0);
  assert(index < rlist->_size);
  return rlist->_omts[index];
}
