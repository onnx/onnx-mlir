/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----------------- Compiler.h - Compiler abstraction support ---------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines several macros, which allows use of compiler-specific
// features in a way that remains portable. This header can be included from
// either C or C++.
//
//===----------------------------------------------------------------------===//

#ifndef OM_COMPILER_MACROS_H
#define OM_COMPILER_MACROS_H

/// ONNX_MLIR_EXPORT - classes, functions, and variables marked with
/// this keywork will be made public and visible outside of any shared library
/// they are linked in to.
#ifdef ONNX_MLIR_BUILT_AS_STATIC
#define ONNX_MLIR_EXPORT
#else
#ifdef _MSC_VER
#ifdef OnnxMlirCompiler_EXPORTS
/* We are building this library */
#define ONNX_MLIR_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define ONNX_MLIR_EXPORT __declspec(dllimport)
#endif
#else
#define ONNX_MLIR_EXPORT __attribute__((__visibility__("default")))
#endif
#endif

#endif
