/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
#ifndef ONNX_MLIR_ONNXMLIRRUNTIME_H
#define ONNX_MLIR_ONNXMLIRRUNTIME_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdbool.h>
#include <stdint.h>
#endif

#include <onnx-mlir/Runtime/OMEntryPoint.h>
#include <onnx-mlir/Runtime/OMInstrument.h>
#include <onnx-mlir/Runtime/OMSignature.h>
#include <onnx-mlir/Runtime/OMTensor.h>
#include <onnx-mlir/Runtime/OMTensorList.h>

/*! \mainpage ONNX-MLIR Runtime API documentation
 *
 * \section intro_sec Introduction
 *
 * ONNX-MLIR project comes with an executable `onnx-mlir` capable
 * of compiling onnx models to a shared library. In this documentation, we
 * demonstrate how to interact programmatically with the compiled
 * shared library using ONNX-MLIR's Runtime API.
 *
 * \section c-runtime-api C Runtime API
 *
 * \subsection data-structures Data Structures
 *
 * `OMTensor` is the data structure used to describe the runtime information
 * (rank, shape, data type, etc) associated with a tensor input or output.
 *
 * `OMTensorList` is the data structure used to hold a list of pointers to
 * OMTensor so that they can be passed into and out of the compiled model as
 * inputs and outputs.
 *
 * \subsection model-entry-point-signature Model Entry Point Signature
 *
 * All compiled models will have the same exact C function signature equivalent
 * to:
 *
 * ```c
 * OMTensorList* run_main_graph(OMTensorList*);
 * ```
 *
 * Intuitively, the model takes a list of tensors as input and returns a list of
 * ensors as output.
 *
 * \subsection invoke-models-using-c-runtime-api Invoke Models Using C Runtime
 * API
 *
 * We demonstrate using the API functions to run a simple ONNX model consisting
 * of an add operation. To create such an onnx model, use this
 * <a href="gen_add_onnx.py" target="_blank"><b>python script</b></a>
 *
 * To compile the above model, run `onnx-mlir add.onnx` and a binary library
 * "add.so" should appear. We can use the following C code to call into the
 * compiled function computing the sum of two inputs:
 *
 * ```c
 * #include <OnnxMlirRuntime.h>
 * #include <stdio.h>
 *
 * OMTensorList *run_main_graph(OMTensorList *);
 *
 * int main() {
 *   // Shared shape & rank.
 *   int64_t shape[] = {3, 2};
 *   int64_t rank = 2;
 *   // Construct x1 omt filled with 1.
 *   float x1Data[] = {1., 1., 1., 1., 1., 1.};
 *   OMTensor *x1 = omTensorCreate(x1Data, shape, rank, ONNX_TYPE_FLOAT);
 *   // Construct x2 omt filled with 2.
 *   float x2Data[] = {2., 2., 2., 2., 2., 2.};
 *   OMTensor *x2 = omTensorCreate(x2Data, shape, rank, ONNX_TYPE_FLOAT);
 *   // Construct a list of omts as input.
 *   OMTensor *list[2] = {x1, x2};
 *   OMTensorList *input = omTensorListCreate(list, 2);
 *   // Call the compiled onnx model function.
 *   OMTensorList *outputList = run_main_graph(input);
 *   // Get the first omt as output.
 *   OMTensor *y = omTensorListGetOmtByIndex(outputList, 0);
 *   float *outputPtr = (float *)omTensorGetDataPtr(y);
 *   // Print its content, should be all 3.
 *   for (int i = 0; i < 6; i++)
 *     printf("%f ", outputPtr[i]);
 *   return 0;
 * }
 * ```
 *
 * Compile with `gcc main.c add.so -o add`, you should see an executable `add`
 * appearing. Run it, and the output should be:
 *
 * ```
 * 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000
 * ```
 * Exactly as it should be.
 *
 * \subsection reference Reference
 *
 * For full reference to available C Runtime API, refer to
 * `include/onnx-mlir/Runtime/OMTensor.h` and
 * `include/onnx-mlir/Runtime/OMTensorList.h`.
 *
 */

#endif // ONNX_MLIR_ONNXMLIRRUNTIME_H
