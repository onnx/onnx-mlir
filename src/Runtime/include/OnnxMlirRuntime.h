//===------- OnnxMlirRuntime.h - ONNX-MLIR Runtime API Declarations -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of external RtMemRef data structures and
// helper functions.
//
//===----------------------------------------------------------------------===//
#ifndef __ONNX_MLIR_RUNTIME_H__
#define __ONNX_MLIR_RUNTIME_H__

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdbool.h>
#include <stdint.h>
#endif

#include <OnnxMlirRuntime/OnnxDataType.h>

/*! \mainpage ONNX-MLIR Model Execution API documentation
 *
 * \section intro_sec Introduction
 *
 * ONNX-MLIR project comes with an executable `onnx-mlir` capable
 * of compiling onnx models to a shared library. In this documentation, we
 * explain and document ways to interact programmatically with the compiled
 * shared library using the API's defined in `src/Runtime/include/OnnxMlir.h`.
 *
 * \section Important Data Structures
 * \subsection RtMemRef
 *
 * `RtMemRef` is the data structure used to describe the runtime information
 * (rank, shape, data type, etc) associated with a tensor (or `MemRef`, in the
 * context of MLIR). Specifically, it can be described as a struct with the
 * following data members:
 *
 * ```cpp
 * void *_data;            // data buffer
 * void *_alignedData;     // aligned data buffer that the rmr indexes.
 * INDEX_TYPE _offset;     // offset of 1st element
 * INDEX_TYPE *_dataSizes; // sizes array
 * int64_t *_dataStrides;  // strides array
 * int _dataType;          // ONNX data type
 * int _rank;              // rank
 * std::string _name;      // optional name for named access
 * bool _owningData;       // indicates whether the Rmr owns the memory space
 *                            referenced by _data. Rmr struct will release the
 * memory space refereced by _data upon destruction if and only if it owns it.
 * ```
 *
 * \subsection RtMemRefList
 * `RtMemRefList` is a data structure used to hold a list of RtMemRef pointers.
 *
 * \section Inference Function Signature
 *
 * All compiled model will have the same exact C function signature equivalent
 * to:
 *
 * ```c
 * RtMemRefList* run_main_graph(RtMemRefList*);
 * ```
 *
 * That is to say, the model inference function consumes a list of input
 * RtMemRefs and produces a list of output RtMemRefs.
 *
 * \section Invoke the Inference Function with C API
 *
 * In this section, we will walk through an example using the API functions to
 *run a simple ONNX model consisting of an add operation. To create such an onnx
 *model, use the following python script:
 *
 * ```py
 * import onnx
 * from onnx import helper
 * from onnx import AttributeProto, TensorProto, GraphProto
 *
 * # Create one input (ValueInfoProto)
 * X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, [3, 2])
 * X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, [3, 2])
 *
 * # Create one output (ValueInfoProto)
 * Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 2])
 *
 * # Create a node (NodeProto) - This is based on Pad-11
 * node_def = helper.make_node(
 *     'Add', # node name
 *     ['X1', 'X2'], # inputs
 *     ['Y'], # outputs
 * )
 *
 * # Create the graph (GraphProto)
 * graph_def = helper.make_graph(
 *     [node_def],
 *     'test-model',
 *     [X1, X2],
 *     [Y],
 * )
 *
 * # Create the model (ModelProto)
 * model_def = helper.make_model(graph_def, producer_name='onnx-example')
 *
 * print('The model is:\n{}'.format(model_def))
 * onnx.checker.check_model(model_def)
 * onnx.save(model_def, "add.onnx")
 * print('The model is checked!')
 *```
 *
 * To compile the above model, run `onnx-mlir add.onnx` and a binary library
 *"add.so" should appear. We can use the following C code to call into the
 *compiled function computing the sum of two inputs:
 *
 * ```c
 * #include <OnnxMlir.h>
 * #include <stdio.h>
 *
 * RtMemRefList *run_main_graph(RtMemRefList *);
 *
 * int main() {
 *   // Construct x1 rmr filled with 1.
 *   float x1Data[] = {1., 1., 1., 1., 1., 1.};
 *   RtMemRef *x1 = rmrCreate(2);
 *   rmrSetData(x1, x1Data);
 *
 *   // Construct x2 rmr filled with 2.
 *   float x2Data[] = {2., 2., 2., 2., 2., 2.};
 *   RtMemRef *x2 = rmrCreate(2);
 *   rmrSetData(x2, x2Data);
 *
 *   // Construct a list of rmrs as input.
 *   RtMemRef *list[2] = {x1, x2};
 *   RtMemRefList *input = rmrListCreate(list, 2);
 *
 *   // Call the compiled onnx model function.
 *   RtMemRefList *outputList = run_main_graph(input);
 *
 *   // Get the first rmr as output.
 *   RtMemRef *y = rmrListGetRmrByIndex(outputList, 0);
 *
 *   // Print its content, should be all 3.
 *   for (int i = 0; i < 6; i++)
 *     printf("%f ", ((float *)rmrGetData(y))[i]);
 *
 *   return 0;
 * }
 * ```
 *
 * Compile with `gcc main.c add.so -o add`, you should see an executable `add` appearing.
 * Run it, and the output should be:
 *
 * ```
 * 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000
 * ```
 * Exactly as it should be.
 */

typedef int64_t INDEX_TYPE;

struct RtMemRef;
typedef struct RtMemRef RtMemRef;

struct RtMemRefList;
typedef struct RtMemRefList RtMemRefList;

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------- */
/* C/C++ API for RtMemRef calls */
/*----------------------------- */

/**
 * RtMemRef creator
 *
 * @param rank, rank of the data sizes and strides
 * @return pointer to RtMemRef created, NULL if creation failed.
 *
 * Create a RtMemRef with specified rank. Memory for data sizes and
 * strides are allocated.
 */
RtMemRef *rmrCreate(int rank);

/**
 * RtMemRef creator
 *
 * @param rank, rank of the data sizes and strides
 * @param name, (optional) name of the tensor
 * @param owningData, whether the rmr owns the underlying data, if true, data
 * pointer will be released when the corresponding rmr gets released or goes out
 * of scope.
 * @return pointer to RtMemRef created, NULL if creation failed.
 *
 * Create a RtMemRef with specified rank, name and data ownership. Memory for
 * data sizes and strides are allocated.
 */
RtMemRef *rmrCreateWithNameAndOwnership(int rank, char *name, bool owningData);

/**
 * RtMemRef destroyer
 *
 * @param rmr, pointer to the RtMemRef
 *
 * Destroy the RtMemRef struct.
 */
void rmrDestroy(RtMemRef *rmr);

/**
 * RtMemRef data getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the data buffer of the RtMemRef,
 *         NULL if the data buffer is not set.
 */
void *rmrGetData(RtMemRef *rmr);

/**
 * RtMemRef data setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param data, data buffer of the RtMemRef to be set
 *
 * Set the data buffer pointer of the RtMemRef. Note that the data buffer
 * is assumed to be managed by the user, i.e., the RtMemRef destructor
 * will not free the data buffer. Because we don't know how exactly the
 * data buffer is allocated, e.g., it could have been allocated on the stack.
 */
void rmrSetData(RtMemRef *rmr, void *data);

/**
 * RtMemRef data sizes getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the data shape array.
 */
INDEX_TYPE *rmrGetDataShape(RtMemRef *rmr);

/**
 * RtMemRef data sizes setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param dataSizes, data sizes array to be set
 *
 * Set the data sizes array of the RtMemRef to the values in the input array.
 */
void rmrSetDataShape(RtMemRef *rmr, INDEX_TYPE *dataSizes);

/**
 * RtMemRef data strides getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the data strides array.
 */
int64_t *rmrGetDataStrides(RtMemRef *rmr);

/**
 * RtMemRef data strides setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param dataStrides, data strides array to be set
 *
 * Set the data strides array of the RtMemRef to the values in the input array.
 */
void rmrSetDataStrides(RtMemRef *rmr, int64_t *dataStrides);

/**
 * RtMemRef data type getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return ONNX data type of the data buffer elements.
 */
int rmrGetDataType(RtMemRef *rmr);

/**
 * RtMemRef data type setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param dataType, ONNX data type to be set
 *
 * Set the ONNX data type of the data buffer elements.
 */
void rmrSetDataType(RtMemRef *rmr, int dataType);

/* Helper function to get the ONNX data type size in bytes */
static inline int getDataTypeSize(int dataType) {
    return dataType < 0 ||
           dataType >= sizeof(RTMEMREF_DATA_TYPE_SIZE) / sizeof(int)
           ? 0
           : RTMEMREF_DATA_TYPE_SIZE[dataType];
}

/**
 * RtMemRef data buffer size getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return the total size of the data buffer in bytes.
 */
 int64_t rmrGetDataBufferSize(RtMemRef *rmr);

/**
 * RtMemRef rank getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return rank of data sizes and strides of the RtMemRef.
 */
int rmrGetRank(RtMemRef *rmr);

/**
 * RtMemRef name getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return pointer to the name of the RtMemRef,
 *         an empty string if the name is not set.
 */
char *rmrGetName(RtMemRef *rmr);

/**
 * RtMemRef name setter
 *
 * @param rmr, pointer to the RtMemRef
 * @param name, name of the RtMemRef to be set
 *
 * Set the name of the RtMemRef.
 */
void rmrSetName(RtMemRef *rmr, char *name);

/**
 * RtMemRef number of elements getter
 *
 * @param rmr, pointer to the RtMemRef
 * @return the number of elements in the data buffer.
 */
INDEX_TYPE rmrGetNumElems(RtMemRef *rmr);

/*---------------------------------------- */
/* C/C++ API for RtMemRefList calls */
/*---------------------------------------- */

/**
 * RtMemRefList creator
 *
 * @param rmrs, array of pointers to RtMemRef
 * @param n, number of elements in rmrs array
 * @return pointer to the RtMemRefList created, NULL if creation failed.
 *
 * Create an RtMemRefList with specified RtMemRef array.
 * If a RtMemRef has a name, in addition to be accessed by its index,
 * the RtMemRef can also be accessed by its name.
 */
RtMemRefList *rmrListCreate(RtMemRef **rmrs, int n);

/**
 * RtMemRefList destroyer
 *
 * @param ormrd, pointer to the RtMemRefList to be destroyed
 *
 * Destroy the RtMemRefList struct.
 */
void rmrListDestroy(RtMemRefList *ormrd);

/**
 * RtMemRefList RtMemRef array getter
 *
 * @param ormrd, pointer to the RtMemRefList
 * @return pointer to the array of RtMemRef pointers.
 */
RtMemRef **rmrListGetPtrToRmrs(RtMemRefList *ormrd);

/**
 * RtMemRefList number of RtMemRefs getter
 *
 * @param ormrd, pointer to the RtMemRefList
 * @return number of elements in the RtMemRef array.
 */
int rmrListGetNumRmrs(RtMemRefList *ormrd);

/**
 * RtMemRefList RtMemRef getter by index
 *
 * @param ormrd, pointer to the RtMemRefList
 * @param index, index of the RtMemRef
 * @reutrn pointer to the RtMemRef, NULL if not found.
 */
RtMemRef *rmrListGetRmrByIndex(RtMemRefList *ormrd, int index);
#ifdef __cplusplus
}
#endif

#endif /* __ONNX_MLIR_RUNTIME_H__ */
