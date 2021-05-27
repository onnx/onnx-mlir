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

#ifdef __APPLE__
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

// Number of shape and strides that are stored directly in the tensor data
// structure. Reduces the number of memory allocation/freeing.
#define OMTENSOR_SUPPORTED_INTERNAL_SIZE 16

// Methods local to this file to initialize and cleanup the lists.
struct OMTensorList;
static int init(OMTensorList *list, size_t size, OMTensor **omts);
static void cleanup(OMTensorList *list, int reset);

struct OMTensorList {
#ifdef __cplusplus
  /**
   * Constructor
   *
   * Create an OMTensorList with specified OMTensor pointer array
   * and the size of the array
   */
  OMTensorList(OMTensor *omts[], int n) {
    if (!init(this, n, omts)) {
      throw std::runtime_error(
          "OMTensorList(" + std::to_string(n) + ") malloc error");
    }
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
  ~OMTensorList() { cleanup(this, false); };
#endif

  /* To facilitate user facing API getOmts, OMTensors are kept in a vector
   * that can be quickly returned as an array. A name to index map is used
   * to address ReMemRefs by name.
   */
  OMTensor **_omts; // OMTensor array
  size_t _size;     // Number of elements in _omts.
  OMTensor *_internalOmts[OMTENSOR_SUPPORTED_INTERNAL_SIZE];
};

// Cleanup the list, which include removing all tensors present in the list.
// With reset on, the fields will be reset (safer if used after cleanup).
static void cleanup(OMTensorList *list, int reset) {
  for (int i = 0; i < list->_size; i++)
    if (list->_omts[i])
      omTensorDestroy(list->_omts[i]);
  if (list->_size > OMTENSOR_SUPPORTED_INTERNAL_SIZE)
    free(list->_omts);
  free(list);
  if (reset) {
    list->_size = 0;
    list->_omts = NULL;
  }
}

// Initialize an OMTensorList with mandatory size and list of tensors. Return 0
// on failure, and cleanup the data structure.
static int init(OMTensorList *list, size_t size, OMTensor **omts) {
  assert(list);
  if (size > 0)
    assert(omts);
  // Initialize fields.
  list->_size = size;
  if (size <= OMTENSOR_SUPPORTED_INTERNAL_SIZE) {
    // When the list can be accommodated in the internal field, do so to avoid
    // malloc/free.
    list->_omts = &(list->_internalOmts[0]);
  } else if (!(list->_omts = (OMTensor **)malloc(sizeof(OMTensor *) * size))) {
    // Failed to alloc, reset and return failure.
    list->_size = 0;
    return 0;
  }
  // Copy pointers (use private list only).
  for (int i = 0; i < list->_size; i++)
    list->_omts[i] = omts[i];
  return 1;
}

/* OMTensorList creator */
OMTensorList *omTensorListCreate(OMTensor **tensors, int n) {
  OMTensorList *list = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  if (!list)
    return NULL;
  if (!init(list, n, tensors)) {
    free(list);
    return NULL;
  }
  return list;
}

/* OMTensorList destroyer */
void omTensorListDestroy(OMTensorList *list) { cleanup(list, true); }

/* OMTensorList OMTensor array getter */
OMTensor **omTensorListGetOmtArray(OMTensorList *list) { return list->_omts; }

/* OMTensorList number of OMTensor getter */
int omTensorListGetSize(OMTensorList *list) { return list->_size; }

/* Return OMTensor at specified index in the OMTensorList */
OMTensor *omTensorListGetOmtByIndex(OMTensorList *list, size_t index) {
  assert(index >= 0);
  assert(index < list->_size);
  return list->_omts[index];
}
