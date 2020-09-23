#ifndef ONNX_MLIR_OMTENSORLIST_H
#define ONNX_MLIR_OMTENSORLIST_H

#include "onnx-mlir/Runtime/OMTensor.h"

struct OMTensorList;

#ifndef __cplusplus
typedef struct OMTensorList OMTensorList;
#endif

/**
 * OMTensorList creator
 *
 * @param tensors array of pointers to OMTensor
 * @param n number of elements in tensors array
 * @return pointer to the OMTensorList created, NULL if creation failed.
 *
 * Create an OMTensorList with specified OMTensor array.
 * If a OMTensor has a name, in addition to be accessed by its index,
 * the OMTensor can also be accessed by its name.
 */
OMTensorList *omTensorListCreate(OMTensor **tensors, int n);

/**
 * OMTensorList destroyer
 *
 * @param list pointer to the OMTensorList to be destroyed
 *
 * Destroy the OMTensorList struct.
 */
void omTensorListDestroy(OMTensorList *list);

/**
 * OMTensorList OMTensor array getter
 *
 * @param list pointer to the OMTensorList
 * @return pointer to the array of OMTensor pointers.
 */
OMTensor **omTensorListGetPtrToOmts(OMTensorList *list);

/**
 * OMTensorList size getter
 *
 * @param list pointer to the OMTensorList
 * @return number of elements in the OMTensor array.
 */
int omTensorListGetSize(OMTensorList *list);

/**
 * OMTensorList OMTensor getter by index
 *
 * @param list pointer to the OMTensorList
 * @param index index of the OMTensor
 * @reutrn pointer to the OMTensor, NULL if not found.
 */
OMTensor *omTensorListGetOmtByIndex(OMTensorList *list, size_t index);

#endif //ONNX_MLIR_OMTENSORLIST_H
