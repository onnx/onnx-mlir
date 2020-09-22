#ifndef ONNX_MLIR_OMTENSORLIST_H
#define ONNX_MLIR_OMTENSORLIST_H

#include "onnx-mlir/Runtime/OMTensor.h"

struct OMTensorList {
#ifdef __cplusplus
    /**
     * Constructor
     *
     * Create an OMTensorList with specified OMTensor pointer array
     * and the size of the array
     */
    OMTensorList(OMTensor *omts[], int n) : _omts(omts), _n(n){};

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
        for (int i = 0; i < _n; i++)
            if (_omts[i])
                omTensorDestroy(_omts[i]);
    };
#endif

    /* To facilitate user facing API getOmts, OMTensors are kept in a vector
     * that can be quickly returned as an array. A name to index map is used
     * to address ReMemRefs by name.
     */
    OMTensor **_omts; // OMTensor array

    size_t _n; // Number of elements in _omts.
};

#ifndef __cplusplus
typedef struct OMTensorList OMTensorList;
#endif


/**
 * OMTensorList creator
 *
 * @param omts, array of pointers to OMTensor
 * @param n, number of elements in omts array
 * @return pointer to the OMTensorList created, NULL if creation failed.
 *
 * Create an OMTensorList with specified OMTensor array.
 * If a OMTensor has a name, in addition to be accessed by its index,
 * the OMTensor can also be accessed by its name.
 */
OMTensorList *omTensorListCreate(OMTensor **omts, int n);

/**
 * OMTensorList destroyer
 *
 * @param list, pointer to the OMTensorList to be destroyed
 *
 * Destroy the OMTensorList struct.
 */
void omTensorListDestroy(OMTensorList *list);

/**
 * OMTensorList OMTensor array getter
 *
 * @param list, pointer to the OMTensorList
 * @return pointer to the array of OMTensor pointers.
 */
OMTensor **omTensorListGetPtrToOmts(OMTensorList *list);

/**
 * OMTensorList number of OMTensors getter
 *
 * @param list, pointer to the OMTensorList
 * @return number of elements in the OMTensor array.
 */
int omTensorListGetNumOmts(OMTensorList *list);

/**
 * OMTensorList OMTensor getter by index
 *
 * @param list, pointer to the OMTensorList
 * @param index, index of the OMTensor
 * @reutrn pointer to the OMTensor, NULL if not found.
 */
OMTensor *omTensorListGetOmtByIndex(OMTensorList *list, int index);

#endif //ONNX_MLIR_OMTENSORLIST_H
