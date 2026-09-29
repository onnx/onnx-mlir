/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMEntryPoint.h - OMEntryPoint Declaration header ---------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMEntryPoint API functions.
//
//===----------------------------------------------------------------------===//
// LLVM-FREE FILE -- DO NOT ADD LLVM / MLIR / ONNX-MLIR COMPILER DEPENDENCES.
//
// Part of the lightweight onnx-mlir build used for the pip-installable
// packages (om_pyrt). Any LLVM/MLIR reference here will break those packages.
//===----------------------------------------------------------------------===//

#include "onnx-mlir/Compiler/OMCompilerMacros.h"

#ifdef __cplusplus
#pragma once

extern "C" {
#endif

/**
 * \brief Return all entry point names in a model. These entry point names are
 * the symbols of the inference functions in the model. Users use them to run
 * inference, e.g. by calling `entry_point_name(OMTensorList).
 *
 * An entry point name can be passed to functions `omInputSignature` and
 * `omOutputSignature` to query its input and output signatures, respectively.
 *
 * The array returned by omQueryEntryPoints is not to be freed because
 * the array is embedded in the compiled model.
 *
 * @param numOfEntryPoints to get the number of entry points.
 * @return an array of strings. The array ends with NULL. For example:
 * ["run_add", "run_sub", NULL].
 */
OM_EXTERNAL_VISIBILITY const char *const *omQueryEntryPoints(
    int64_t *numOfEntryPoints);

#ifdef __cplusplus
}
#endif
