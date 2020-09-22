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

#include <onnx-mlir/Runtime/OMTensor.h>
#include <onnx-mlir/Runtime/OMTensorList.h>

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
 * void *_alignedData;     // aligned data buffer that the omt indexes.
 * int64_t _offset;     // offset of 1st element
 * int64_t *_dataSizes; // sizes array
 * int64_t *_dataStrides;  // strides array
 * int _dataType;          // ONNX data type
 * int _rank;              // rank
 * std::string _name;      // optional name for named access
 * bool _owningData;       // indicates whether the Omt owns the memory space
 *                            referenced by _data. Omt struct will release the
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
 *   // Construct x1 omt filled with 1.
 *   float x1Data[] = {1., 1., 1., 1., 1., 1.};
 *   OMTensor *x1 = omTensorCreateEmpty(2);
 *   omTensorSetData(x1, x1Data);
 *
 *   // Construct x2 omt filled with 2.
 *   float x2Data[] = {2., 2., 2., 2., 2., 2.};
 *   OMTensor *x2 = omTensorCreateEmpty(2);
 *   omTensorSetData(x2, x2Data);
 *
 *   // Construct a list of omts as input.
 *   OMTensor *list[2] = {x1, x2};
 *   OMTensorList *input = omTensorListCreate(list, 2);
 *
 *   // Call the compiled onnx model function.
 *   OMTensorList *outputList = run_main_graph(input);
 *
 *   // Get the first omt as output.
 *   OMTensor *y = omTensorListGetOmtByIndex(outputList, 0);
 *
 *   // Print its content, should be all 3.
 *   for (int i = 0; i < 6; i++)
 *     printf("%f ", ((float *)omTensorGetData(y))[i]);
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

#endif /* __ONNX_MLIR_RUNTIME_H__ */
