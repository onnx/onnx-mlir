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

struct OMTensorList {
#ifdef __cplusplus

  /**
   * Constructor
   *
   * Create an OMTensorList with specified OMTensor pointer array, the size of
   * the array and indication of whether or not to perform a shallow destroy of
   * the OMTensorList
   */
  OMTensorList(OMTensor *omts[], int64_t n) : _omts(omts), _size(n) {
    _shallow = false;
  };

  /**
   * Destructor
   *
   * Destroy the OMTensorList struct.
   */
  ~OMTensorList() {
    if (_shallow)
      omTensorListDestroyShallow(this);
    else
      /* Destroy all the OMTensors */
      omTensorListDestroy(this);
    free(_omts);
  };
#endif

  /* To facilitate user facing API getOmts, OMTensors are kept in a vector
   * that can be quickly returned as an array. A name to index map is used
   * to address ReMemRefs by name.
   */
  OMTensor **_omts; // OMTensor array

  int64_t _size; // Number of elements in _omts.

  int64_t
      _shallow; // indicates whether we want to perform a shallow destroy which
                // means the tensors are not destroyed but rather the list as
                // well as the pointer to the OMTensor array are freed.
};

/* OMTensorList creator */
OMTensorList *omTensorListCreate(OMTensor **tensors, int64_t n) {
  return omTensorListCreateWithShallow(tensors, n, false);
}

/* OMTensorList creator with shallow destructor */
OMTensorList *omTensorListCreateWithShallow(
    OMTensor **tensors, int64_t n, int64_t shallow) {
  OMTensorList *list = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  if (!list)
    return NULL;
  list->_size = n;
  list->_shallow = shallow;
  list->_omts = (OMTensor **)malloc(sizeof(OMTensor *) * n);

  if (!list->_omts) {
    free(list); // free the previously allocated memory in case of an error
    return NULL;
  }
  // copy the pointers to the OMTensor array
  // now OMTensorList will always own the tensor array of pointers
  memcpy(tensors, list->_omts, list->_size);

  return list;
}

/* OMTensorList destroyer */
void omTensorListDestroy(OMTensorList *list) {
  if (!list)
    return;
  for (int64_t i = 0; i < list->_size; i++)
    omTensorDestroy(list->_omts[i]);
  // Free the list as well as the pointers to the OMTensor array
  omTensorListDestroyShallow(list);
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
OMTensor **omTensorListGetOmtArray(OMTensorList *list) { return list->_omts; }

/* OMTensorList number of OMTensor getter */
int64_t omTensorListGetSize(OMTensorList *list) { return list->_size; }

/* Return OMTensor at specified index in the OMTensorList */
OMTensor *omTensorListGetOmtByIndex(OMTensorList *rlist, int64_t index) {
  assert(index >= 0);
  assert(index < rlist->_size);
  return rlist->_omts[index];
}
