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
#include <stdlib.h>
#endif

#ifdef __cplusplus
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#else
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
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
  if (list->_owning)
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

/* Helper function to create a sub-string between the start and end strings */
static inline char *createSubStr(
    char *str, const char *start, const char *end) {
  char *startPos = strstr(str, start);
  char *endPos = strstr(str, end);
  char *p;
  return (startPos && endPos && (p = startPos + strlen(start)) < endPos)
             ? strndup(p, endPos - p)
             : NULL;
}

/**
 * Verify OMTensorList again the given signature
 * Signature must be in JSON format with keys `type`, `dims` and `name`, e.g.
 *   [{"type" : "f32" , "dims" : [4, 8, 16] , "name": "t1"},
 *    {"type" : "i32" , "dims" : [32, 64, 128] , "name": "t2"}]
 */
int64_t omTensorListVerifyType(OMTensorList *list, const char *sig) {
  int64_t passed = 1, failed = 0;
  int64_t actualNumTensors = omTensorListGetSize(list);

  int64_t sigLen = strlen(sig);
  char *sigStr = (char *)malloc(sigLen + 1);
  // Copy and remove spaces, double-quotes and newlines.
  int64_t j = 0;
  for (int64_t i = 0; i < sigLen; ++i) {
    if ((sig[i] == ' ') || (sig[i] == '"') || (sig[i] == '\n'))
      continue;
    sigStr[j++] = sig[i];
  }
  // If the new string is shorter, append spaces to the end to avoid corrupted
  // memory when freeing memory.
  for (int64_t k = j; k < sigLen; ++k)
    sigStr[k] = ' ';
  sigStr[sigLen] = '\0';

  // Process the signature to extract type info for each input tensor.
  // The type info will be used to verify user input tensors.
  int64_t refNumTensors = 0;
  char *nextStart = strchr(sigStr, '{');
  while (nextStart != NULL) {
    char *nextEnd = strchr(nextStart, '}');
    if ((nextEnd == NULL) || (nextStart >= nextEnd)) {
      // { and } in the signature do not match or empty info. Cannot verify.
      free(sigStr);
      return passed;
    }
    char *refTensorStr = createSubStr(nextStart, "{", "}");
    assert(refTensorStr && "Something really wrong");
    // Do not support nested structure. Just let it pass.
    if (strchr(refTensorStr, '{') != NULL) {
      free(refTensorStr);
      free(sigStr);
      return passed;
    }

    if (refNumTensors >= actualNumTensors) {
      printf("Wrong number of input tensors: expect more input tensors.\n");
      free(refTensorStr);
      free(sigStr);
      errno = EINVAL;
      return failed;
    }

    // Get type info from the tensor passed by users.
    OMTensor *tensor = omTensorListGetOmtByIndex(list, refNumTensors);
    OM_DATA_TYPE actualType = omTensorGetDataType(tensor);
    int64_t actualRank = omTensorGetRank(tensor);
    int64_t *actualShape = omTensorGetShape(tensor);

    // Verify data type.
    char *refTypeStr = createSubStr(refTensorStr, "type:", ",");
    if (refTypeStr != NULL) {
      // Type check is passed by default and we will handle cases we know.
      bool typeIsOK = true;
      switch (actualType) {
      case ONNX_TYPE_FLOAT:
        typeIsOK = (strcmp(refTypeStr, "f32") == 0);
        break;
      case ONNX_TYPE_UINT8:
        typeIsOK = (strcmp(refTypeStr, "ui8") == 0);
        break;
      case ONNX_TYPE_INT8:
        typeIsOK = (strcmp(refTypeStr, "i8") == 0);
        break;
      case ONNX_TYPE_UINT16:
        typeIsOK = (strcmp(refTypeStr, "ui16") == 0);
        break;
      case ONNX_TYPE_INT16:
        typeIsOK = (strcmp(refTypeStr, "i16") == 0);
        break;
      case ONNX_TYPE_UINT32:
        typeIsOK = (strcmp(refTypeStr, "ui32") == 0);
        break;
      case ONNX_TYPE_INT32:
        typeIsOK = (strcmp(refTypeStr, "i32") == 0);
        break;
      case ONNX_TYPE_UINT64:
        typeIsOK = (strcmp(refTypeStr, "ui64") == 0);
        break;
      case ONNX_TYPE_INT64:
        typeIsOK = (strcmp(refTypeStr, "i64") == 0);
        break;
      case ONNX_TYPE_STRING:
        typeIsOK = true; // no verification at this moment.
        break;
      case ONNX_TYPE_BOOL:
        typeIsOK = (strcmp(refTypeStr, "i1") == 0);
        break;
      case ONNX_TYPE_FLOAT16:
        typeIsOK = (strcmp(refTypeStr, "f16") == 0);
        break;
      case ONNX_TYPE_DOUBLE:
        typeIsOK = (strcmp(refTypeStr, "f64") == 0);
        break;
      case ONNX_TYPE_COMPLEX64:
        typeIsOK = true; // no verification at this moment.
        break;
      case ONNX_TYPE_COMPLEX128:
        typeIsOK = true; // no verification at this moment.
        break;
      case ONNX_TYPE_BFLOAT16:
        typeIsOK = true; // no verification at this moment.
        break;
      case ONNX_TYPE_UNDEFINED:
        typeIsOK = false;
        break;
      }
      if (!typeIsOK) {
        printf("Wrong data type for the input %lld: expect %s, but got %s.\n",
            (long long)refNumTensors, refTypeStr,
            OM_DATA_TYPE_NAME[actualType]);
        free(refTypeStr);
        free(refTensorStr);
        free(sigStr);
        errno = EINVAL;
        return failed;
      }
      free(refTypeStr);
    }

    // Verify data shape.
    char *refShapeStr = createSubStr(refTensorStr, "dims:[", "],");
    if (refShapeStr != NULL) {
      int64_t refRank = 0;
      char *dim = strtok(refShapeStr, ",");
      while (dim != NULL) {
        if (refRank >= actualRank) {
          printf("Wrong rank for the input %lld: expect more dimensions.\n",
              (long long)refNumTensors);
          free(refShapeStr);
          free(refTensorStr);
          free(sigStr);
          errno = EINVAL;
          return failed;
        }
        int refDimSize = atoi(dim);
        if (refDimSize != -1 && refDimSize != actualShape[refRank]) {
          printf("Wrong size for the dimension %lld of the input %lld: expect "
                 "%ld, but got %lld.\n",
              (long long)refRank, (long long)refNumTensors, (long)refDimSize,
              (long long)actualShape[refRank]);
          free(refShapeStr);
          free(refTensorStr);
          free(sigStr);
          errno = EINVAL;
          return failed;
        }
        dim = strtok(NULL, ",");
        refRank++;
      }
      if (refRank != actualRank) {
        printf("Wrong rank for the input %lld: expect %lld, but got %lld.\n",
            (long long)refNumTensors, (long long)refRank,
            (long long)actualRank);
        free(refShapeStr);
        free(refTensorStr);
        free(sigStr);
        errno = EINVAL;
        return failed;
      }
      free(refShapeStr);
    }

    free(refTensorStr);
    refNumTensors++;
    nextStart = strchr(nextEnd, '{');
  }

  // Verify the number of input tensors.
  if (actualNumTensors != refNumTensors) {
    printf("Wrong number of input tensors: expect %lld, but got %lld.\n",
        (long long)refNumTensors, (long long)actualNumTensors);
    free(sigStr);
    errno = EINVAL;
    return failed;
  }

  free(sigStr);
  return passed;
}
