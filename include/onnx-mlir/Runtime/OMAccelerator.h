/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMAccelerator.h ---------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Accelerator base class
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ACCELERATORS_H
#define ONNX_MLIR_ACCELERATORS_H

#include "src/Accelerators/Accelerators.inc"

// Define the macros used to generate various accelerators artifacts (via the
// use of the APPLY_TO_ACCELERATORS macro, which is defined in the cmake
// generated file Accelerators.inc).
#define CREATE_ACCEL_ENUM(name) OM_ACCEL_##name,
#define DECLARE_ACCEL_INIT_FUNCTION(name) extern void OMInitAccel##name();
#define DECLARE_ACCEL_SHUTDOWN_FUNCTION(name) extern void OMShutdownAccel##name();
#define DECLARE_ACCEL_IS_INIT_VARIABLE(name) extern long OMIsInitAccel##name;
#define DEFINE_ACCEL_IS_INIT_VARIABLE(name) long OMIsInitAccel##name = 0;

#ifdef __cplusplus
extern "C" {
#endif

// Define the types of accelerators.
enum OM_DATA_TYPE {
  APPLY_TO_ACCELERATORS(CREATE_ACCEL_ENUM)
  OM_ACCEL_NONE /* no accelerator, aka CPU */
};

// Define runtime init functions for the accelerators that are defined.
// Function is defined as OMInitAccelX where X is the name of the accelerator,
// for example void ONInitAccelNNPA();
// This function is implicitly called when first encountering the computation of
// an onnx graph, but it can also be explicitely called. Function is thread safe
// and can be called arbitrarily many times; it has internal logic to effectively
// initialize the device only once.
APPLY_TO_ACCELERATORS(DECLARE_ACCEL_INIT_FUNCTION)

// Define runtime shutdown functions for the accelerators that are defined.
// This function cannot be called while a onnx graph is being computed. It is
// the responsability of the user to call this function only once all computations
// have completed. Failure to respect this condition will result in undefined 
// onnx graph evaluation's result.
APPLY_TO_ACCELERATORS(DECLARE_ACCEL_SHUTDOWN_FUNCTION)

// Define the variable that records if an accelerator is initialized (nonzero)
// or is uninitialized (zero). Value is initially zero, can only be set in
// a mutex, but can be read without mutex.
APPLY_TO_ACCELERATORS(DECLARE_ACCEL_IS_INIT_VARIABLE)

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_ACCELERATORS_H