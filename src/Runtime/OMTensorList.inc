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

/* OMTensorList creator */
OMTensorList *omTensorListCreate(OMTensor **omts, int n) {
  OMTensorList *list = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  list->_omts = omts;
  list->_n = n;
  return list;
}

/* OMTensorList destroyer */
void omTensorListDestroy(OMTensorList *list) {
  for (int i = 0; i < list->_n; i++)
    free(list->_omts[i]);
  free(list);
}

/* OMTensorList OMTensor array getter */
OMTensor **omTensorListGetPtrToOmts(OMTensorList *list) { return list->_omts; }

/* OMTensorList number of OMTensor getter */
int omTensorListGetNumOmts(OMTensorList *list) { return list->_n; }

/* Return OMTensor at specified index in the OMTensorList */
OMTensor *omTensorListGetOmtByIndex(OMTensorList *rlist, int index) {
  assert(index >= 0);
  assert(index < rlist->_n);
  return rlist->_omts[index];
}