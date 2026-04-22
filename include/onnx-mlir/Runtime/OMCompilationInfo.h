/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- OMCompilationInfo.h - OMCompilationInfo Declaration header -------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMCompilationInfo API function.
//
//===----------------------------------------------------------------------===//

#include "onnx-mlir/Compiler/OMCompilerMacros.h"

#ifdef __cplusplus
#pragma once

extern "C" {
#endif

/**
 * \brief Return the compilation information of the model as a JSON string.
 *
 * The compilation information includes compile options and operation statistics
 * used during model compilation. The format is:
 * {"compile_options": "<string>", "op_stats": <json_object>}
 *
 * The string returned by omCompilationInfo does not have to be freed because
 * it is a part of the model.
 *
 * @return pointer to compilation information JSON string
 */
OM_EXTERNAL_VISIBILITY const char *omCompilationInfo(void);

#ifdef __cplusplus
}
#endif
