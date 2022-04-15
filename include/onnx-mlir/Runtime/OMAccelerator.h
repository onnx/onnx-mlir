/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMAccelerator.h ---------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Header for Onnx Mlir Runtime variables/functions related to accelerators.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ACCELERATORS_H
#define ONNX_MLIR_ACCELERATORS_H

#include "src/Accelerators/Accelerators.inc"

// Define the macros used to generate various accelerators artifacts (via the
// use of the APPLY_TO_ACCELERATORS macro, which is defined in the cmake
// generated file Accelerators.inc).
#define DECLARE_OM_ACCEL_INIT_FUNCTION(name)                                   \
  OM_EXTERNAL_VISIBILITY void OMInitAccel##name();
#define DECLARE_OM_ACCEL_SHUTDOWN_FUNCTION(name)                               \
  OM_EXTERNAL_VISIBILITY void OMShutdownAccel##name();

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Initialize a specific accelerator.
 *
 * Function is defined as OMInitAccelX where X is the name of the accelerator;
 * for the the X=NNPA accelerator, the call is ONInitAccelNNPA().
 *
 * This function is implicitly called before the evaluation of each
 * onnx graph, but it can also be explicitely called. Function is thread safe
 * and can be called arbitrarily many times. The function has internal logic to
 * effectively initialize the device only once.
 *
 */
APPLY_TO_ACCELERATORS(DECLARE_OM_ACCEL_INIT_FUNCTION)

/**
 * \brief Shutdown a specific accelerator.
 *
 * Function is defined as OMShutdownAccelX where X is the name of the
 * accelerator; for the the X=NNPA accelerator, the call is
 * ONShutdownAccelNNPA().
 *
 * This function cannot be called while a onnx graph is being evaluated. It is
 * the responsability of the user to call this function only once all
 * evaluations have completed. Failure to respect this condition will result in
 * undefined onnx graph evaluation's result.
 *
 * Function is thread safe and can be called arbitrarily many times. The
 * function has internal logic to effectively shutdown the device only once.
 *
 */
APPLY_TO_ACCELERATORS(DECLARE_OM_ACCEL_SHUTDOWN_FUNCTION)

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_ACCELERATORS_H