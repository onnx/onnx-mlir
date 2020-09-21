//===------- OnnxMlirRuntime.h - ONNX-MLIR Runtime API Declarations -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of external OMTensor data structures and
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
 * \subsection OMTensor
 *
 * `OMTensor` is the data structure used to describe the runtime information
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
 * \subsection OMTensorList
 * `OMTensorList` is a data structure used to hold a list of OMTensor pointers.
 *
 * \section Inference Function Signature
 *
 * All compiled model will have the same exact C function signature equivalent
 * to:
 *
 * ```c
 * OMTensorList* run_main_graph(OMTensorList*);
 * ```
 *
 * That is to say, the model inference function consumes a list of input
 * OMTensors and produces a list of output OMTensors.
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
 * OMTensorList *run_main_graph(OMTensorList *);
 *
 * int main() {
 *   // Construct x1 rmr filled with 1.
 *   float x1Data[] = {1., 1., 1., 1., 1., 1.};
 *   OMTensor *x1 = rmrCreate(2);
 *   rmrSetData(x1, x1Data);
 *
 *   // Construct x2 rmr filled with 2.
 *   float x2Data[] = {2., 2., 2., 2., 2., 2.};
 *   OMTensor *x2 = rmrCreate(2);
 *   rmrSetData(x2, x2Data);
 *
 *   // Construct a list of rmrs as input.
 *   OMTensor *list[2] = {x1, x2};
 *   OMTensorList *input = rmrListCreate(list, 2);
 *
 *   // Call the compiled onnx model function.
 *   OMTensorList *outputList = run_main_graph(input);
 *
 *   // Get the first rmr as output.
 *   OMTensor *y = rmrListGetRmrByIndex(outputList, 0);
 *
 *   // Print its content, should be all 3.
 *   for (int i = 0; i < 6; i++)
 *     printf("%f ", ((float *)rmrGetData(y))[i]);
 *
 *   return 0;
 * }
 * ```
 *
 * Compile with `gcc main.c add.so -o add`, you should see an executable `add`
 *appearing. Run it, and the output should be:
 *
 * ```
 * 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000
 * ```
 * Exactly as it should be.
 */

typedef int64_t INDEX_TYPE;

struct OMTensor;
typedef struct OMTensor OMTensor;

struct OMTensorList;
typedef struct OMTensorList OMTensorList;

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------- */
/* C/C++ API for OMTensor calls */
/*----------------------------- */

/**
 * OMTensor creator
 *
 * @param rank, rank of the data sizes and strides
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 * Create a OMTensor with specified rank. Memory for data sizes and
 * strides are allocated.
 */
OMTensor *rmrCreate(int rank);

/**
 * OMTensor creator
 *
 * @param rank, rank of the data sizes and strides
 * @param name, (optional) name of the tensor
 * @param owningData, whether the rmr owns the underlying data, if true, data
 * pointer will be released when the corresponding rmr gets released or goes out
 * of scope.
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 * Create a OMTensor with specified rank, name and data ownership. Memory for
 * data sizes and strides are allocated.
 */
OMTensor *rmrCreateWithNameAndOwnership(int rank, char *name, bool owningData);

/**
 * OMTensor destroyer
 *
 * @param rmr, pointer to the OMTensor
 *
 * Destroy the OMTensor struct.
 */
void rmrDestroy(OMTensor *rmr);

/**
 * OMTensor data getter
 *
 * @param rmr, pointer to the OMTensor
 * @return pointer to the data buffer of the OMTensor,
 *         NULL if the data buffer is not set.
 */
void *rmrGetData(OMTensor *rmr);

/**
 * OMTensor data setter
 *
 * @param rmr, pointer to the OMTensor
 * @param data, data buffer of the OMTensor to be set
 *
 * Set the data buffer pointer of the OMTensor. Note that the data buffer
 * is assumed to be managed by the user, i.e., the OMTensor destructor
 * will not free the data buffer. Because we don't know how exactly the
 * data buffer is allocated, e.g., it could have been allocated on the stack.
 */
void rmrSetData(OMTensor *rmr, void *data);

/**
 * OMTensor data sizes getter
 *
 * @param rmr, pointer to the OMTensor
 * @return pointer to the data shape array.
 */
INDEX_TYPE *rmrGetDataShape(OMTensor *rmr);

/**
 * OMTensor data sizes setter
 *
 * @param rmr, pointer to the OMTensor
 * @param dataSizes, data sizes array to be set
 *
 * Set the data sizes array of the OMTensor to the values in the input array.
 */
void rmrSetDataShape(OMTensor *rmr, INDEX_TYPE *dataSizes);

/**
 * OMTensor data strides getter
 *
 * @param rmr, pointer to the OMTensor
 * @return pointer to the data strides array.
 */
int64_t *rmrGetDataStrides(OMTensor *rmr);

/**
 * OMTensor data strides setter
 *
 * @param rmr, pointer to the OMTensor
 * @param dataStrides, data strides array to be set
 *
 * Set the data strides array of the OMTensor to the values in the input array.
 */
void rmrSetDataStrides(OMTensor *rmr, int64_t *dataStrides);

/**
 * OMTensor data type getter
 *
 * @param rmr, pointer to the OMTensor
 * @return ONNX data type of the data buffer elements.
 */
int rmrGetDataType(OMTensor *rmr);

/**
 * OMTensor data type setter
 *
 * @param rmr, pointer to the OMTensor
 * @param dataType, ONNX data type to be set
 *
 * Set the ONNX data type of the data buffer elements.
 */
void rmrSetDataType(OMTensor *rmr, int dataType);

/* Helper function to get the ONNX data type size in bytes */
static inline int getDataTypeSize(int dataType) {
  return dataType < 0 ||
                 dataType >= sizeof(RTMEMREF_DATA_TYPE_SIZE) / sizeof(int)
             ? 0
             : RTMEMREF_DATA_TYPE_SIZE[dataType];
}

/**
 * OMTensor data buffer size getter
 *
 * @param rmr, pointer to the OMTensor
 * @return the total size of the data buffer in bytes.
 */
int64_t rmrGetDataBufferSize(OMTensor *rmr);

/**
 * OMTensor rank getter
 *
 * @param rmr, pointer to the OMTensor
 * @return rank of data sizes and strides of the OMTensor.
 */
int rmrGetRank(OMTensor *rmr);

/**
 * OMTensor name getter
 *
 * @param rmr, pointer to the OMTensor
 * @return pointer to the name of the OMTensor,
 *         an empty string if the name is not set.
 */
char *rmrGetName(OMTensor *rmr);

/**
 * OMTensor name setter
 *
 * @param rmr, pointer to the OMTensor
 * @param name, name of the OMTensor to be set
 *
 * Set the name of the OMTensor.
 */
void rmrSetName(OMTensor *rmr, char *name);

/**
 * OMTensor number of elements getter
 *
 * @param rmr, pointer to the OMTensor
 * @return the number of elements in the data buffer.
 */
INDEX_TYPE rmrGetNumElems(OMTensor *rmr);

/*---------------------------------------- */
/* C/C++ API for OMTensorList calls */
/*---------------------------------------- */

/**
 * OMTensorList creator
 *
 * @param rmrs, array of pointers to OMTensor
 * @param n, number of elements in rmrs array
 * @return pointer to the OMTensorList created, NULL if creation failed.
 *
 * Create an OMTensorList with specified OMTensor array.
 * If a OMTensor has a name, in addition to be accessed by its index,
 * the OMTensor can also be accessed by its name.
 */
OMTensorList *rmrListCreate(OMTensor **rmrs, int n);

/**
 * OMTensorList destroyer
 *
 * @param ormrd, pointer to the OMTensorList to be destroyed
 *
 * Destroy the OMTensorList struct.
 */
void rmrListDestroy(OMTensorList *ormrd);

/**
 * OMTensorList OMTensor array getter
 *
 * @param ormrd, pointer to the OMTensorList
 * @return pointer to the array of OMTensor pointers.
 */
OMTensor **rmrListGetPtrToRmrs(OMTensorList *ormrd);

/**
 * OMTensorList number of OMTensors getter
 *
 * @param ormrd, pointer to the OMTensorList
 * @return number of elements in the OMTensor array.
 */
int rmrListGetNumRmrs(OMTensorList *ormrd);

/**
 * OMTensorList OMTensor getter by index
 *
 * @param ormrd, pointer to the OMTensorList
 * @param index, index of the OMTensor
 * @reutrn pointer to the OMTensor, NULL if not found.
 */
OMTensor *rmrListGetRmrByIndex(OMTensorList *ormrd, int index);
#ifdef __cplusplus
}
#endif

#endif /* __ONNX_MLIR_RUNTIME_H__ */
