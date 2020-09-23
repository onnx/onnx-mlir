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

struct OMTensorList {
#ifdef __cplusplus
  /**
   * Constructor
   *
   * Create an OMTensorList with specified OMTensor pointer array
   * and the size of the array
   */
  OMTensorList(OMTensor *omts[], int n) : _omts(omts), _size(n){};

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
    for (int i = 0; i < _size; i++)
      if (_omts[i])
        omTensorDestroy(_omts[i]);
  };
#endif

  /* To facilitate user facing API getOmts, OMTensors are kept in a vector
   * that can be quickly returned as an array. A name to index map is used
   * to address ReMemRefs by name.
   */
  OMTensor **_omts; // OMTensor array

  size_t _size; // Number of elements in _omts.
};

/* OMTensorList creator */
OMTensorList *omTensorListCreate(OMTensor **tensors, int n) {
  OMTensorList *list = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  list->_omts = tensors;
  list->_size = n;
  return list;
}

/* OMTensorList destroyer */
void omTensorListDestroy(OMTensorList *list) {
  for (int i = 0; i < list->_size; i++)
    free(list->_omts[i]);
  free(list);
}

/* OMTensorList OMTensor array getter */
OMTensor **omTensorListGetPtrToOmts(OMTensorList *list) { return list->_omts; }

/* OMTensorList number of OMTensor getter */
int omTensorListGetSize(OMTensorList *list) { return list->_size; }

/* Return OMTensor at specified index in the OMTensorList */
OMTensor *omTensorListGetOmtByIndex(OMTensorList *rlist, size_t index) {
  assert(index >= 0);
  assert(index < rlist->_size);
  return rlist->_omts[index];
}