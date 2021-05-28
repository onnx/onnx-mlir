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

#include <stdio.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensorList.h"

// TODO In debug mode (which we should turn off eventually) we don't exit on
// errors.
#define DEBUG 1

// Number of shape and strides that are stored directly in the tensor data
// structure. Reduces the number of memory allocation/freeing.
#define OMTENSOR_SUPPORTED_INTERNAL_SIZE 16

// Methods local to this file to initialize and cleanup the lists.
struct OMTensorList;
static bool init(OMTensorList *list, size_t size, int owning, OMTensor **omts);
static void cleanup(OMTensorList *list);

struct OMTensorList {
#ifdef __cplusplus
  /**
   * Constructor
   *
   * Create an OMTensorList with specified OMTensor pointer array
   * and the size of the array
   */
  OMTensorList(OMTensor *omts[], int n) {
    if (!init(this, n, /*owning*/ false, omts)) {
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
  ~OMTensorList() { cleanup(this); };
#endif

  /* To facilitate user facing API getOmts, OMTensors are kept in a vector
   * that can be quickly returned as an array. A name to index map is used
   * to address ReMemRefs by name.
   */
  OMTensor **_omts; // OMTensor array
  size_t _size;     // Number of elements in _omts.
  int _owning;      // indicates whether the OMTensorList owns the pointer to
               // OMTensor array or not. OMTensorList struct will release the
               // memory space referred to by '_omts' upon destruction if and
               // only if it owns it.
  OMTensor *_internalOmts[OMTENSOR_SUPPORTED_INTERNAL_SIZE];
};

// Cleanup the list, which include removing all tensors present in the list.
// With reset on, the fields will be reset (safer if used after cleanup).
static void cleanup(OMTensorList *list) {
  for (int i = 0; i < list->_size; i++)
    if (list->_omts[i])
      omTensorDestroy(list->_omts[i]);
  if (list->_owning || list->_size > OMTENSOR_SUPPORTED_INTERNAL_SIZE)
    free(list->_omts);
}

// Initialize an OMTensorList with mandatory size and list of tensors. Return 0
// on failure, and cleanup the data structure.
static bool init(OMTensorList *list, size_t size, int owning, OMTensor **omts) {
  if (size > 0 && !omts) {
    fprintf(stderr,
        "Error: with positive sizes, expect a nonnull list of tensors\n");
    if (DEBUG)
      return false;
    exit(1);
  }
  // Initialize fields.
  list->_size = size;
  list->_owning = owning;
  if (owning) {
    // no copying, just reuse the list
    list->_omts = omts;
  } else {
    // will copy the list
    if (size <= OMTENSOR_SUPPORTED_INTERNAL_SIZE) {
      // When the list can be accommodated in the internal field, do so to avoid
      // malloc/free.
      list->_omts = &(list->_internalOmts[0]);
    } else if (!(list->_omts =
                       (OMTensor **)malloc(sizeof(OMTensor *) * size))) {
      // Failed to alloc, reset and return failure.
      list->_size = 0;
      return false;
    }
    // Copy pointers (use private list only).
    memcpy(list->_omts, omts, size * sizeof(OMTensor *));
  }
  return true;
}

/* OMTensorList creator */
OMTensorList *omTensorListCreate(OMTensor **tensors, int n) {
  OMTensorList *list = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  if (!list)
    return NULL;
  if (!init(list, n, /*owning*/ false, tensors)) {
    free(list);
    return NULL;
  }
  return list;
}

/* OMTensorList creator with ownership */
OMTensorList *omTensorListCreateWithOwnership(
    OMTensor **tensors, int n, int owning) {
  OMTensorList *list = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  if (!list)
    return NULL;
  if (!init(list, n, owning, tensors)) {
    free(list);
    return NULL;
  }
  return list;
}

/* OMTensorList destroyer */
void omTensorListDestroy(OMTensorList *list) {
  cleanup(list);
  free(list);
}

/* OMTensorList OMTensor array getter */
OMTensor **omTensorListGetOmtArray(OMTensorList *list) {
  if (!list) {
    fprintf(stderr, "Error: expected nonnull list to query\n");
    if (DEBUG)
      return NULL;
    exit(1);
  }
  return list->_omts;
}

/* OMTensorList number of OMTensor getter */
int omTensorListGetSize(OMTensorList *list) {
  if (!list) {
    fprintf(stderr, "Error: expected nonnull list to query\n");
    if (DEBUG)
      return 0;
    exit(1);
  }
  return list->_size;
}

/* Return OMTensor at specified index in the OMTensorList */
OMTensor *omTensorListGetOmtByIndex(OMTensorList *list, size_t index) {
  if (!list) {
    fprintf(stderr, "Error: expected nonnull list to query\n");
    if (DEBUG)
      return NULL;
    exit(1);
  }
  if (index < 0 || index >= list->_size)
    return NULL;
  return list->_omts[index];
}
