/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMSignature.h - OMSignature Declaration header -----------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMSignature API functions.
//
//===----------------------------------------------------------------------===//

#include "onnx-mlir/Compiler/OMCompilerMacros.h"

#ifdef __cplusplus
#pragma once

extern "C" {
#endif

/**
 * \brief Return the input signature of the given entry point as a JSON string.
 *
 * The array returned by omInputSignature does not have to be freed because
 * it is a part of the model.
 *
 * @param entry point name
 * @return pointer to input signature JSON string
 */
OM_EXTERNAL_VISIBILITY const char *omInputSignature(const char *entryPointName);

/**
 * \brief Return the output signature of the given entry point as a JSON string.
 *
 * The array returned by omOutputSignature is not to be freed because
 * the array is embedded in the compiled model.
 *
 * @param entry point name
 * @return pointer to output signature JSON string
 */
OM_EXTERNAL_VISIBILITY const char *omOutputSignature(
    const char *entryPointName);

#ifdef __cplusplus
}
#endif
