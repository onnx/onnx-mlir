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
 * \brief Return the compilation information of the model as a JSON string.
 *
 * The compilation information includes compile options and operation statistics
 * used during model compilation. The format is:
 * {
 *    "compiler_version": "<string>",
 *    "compile_options": "<string>",
 *    "op_stats": <json_object>
 *  }
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
