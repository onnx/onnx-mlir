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
  //  try {
  //    return new OMTensorList(omts, n);
  //  } catch (const invalid_argument &e) {
  //    return NULL;
  //  }
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

#ifdef __cplusplus
/* Return OMTensor of specified name in the OMTensorList */
OMTensor *omTensorListGetOmtByName(OMTensorList *rlist, std::string name) {
  for (int i = 0; i < rlist->_n; i++)
    if (rlist->_omts[i]->_name == name)
      return rlist->_omts[i];
  return NULL;
}
#endif // __cplusplus