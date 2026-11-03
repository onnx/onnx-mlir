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

#ifndef ONNX_MLIR_COMPILER_MACROS_H
#define ONNX_MLIR_COMPILER_MACROS_H

/// OM_EXTERNAL_VISIBILITY - classes, functions, and variables marked with this
/// keyword will be made public and visible outside of any shared library they
/// are linked in to. This is needed when LLVM cmake propagates
/// -fvisibility=hidden via llvm_update_compile_flags.
#if defined(_WIN32)
#define OM_EXTERNAL_VISIBILITY __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define OM_EXTERNAL_VISIBILITY __attribute__((visibility("default")))
#else
#define OM_EXTERNAL_VISIBILITY
#endif

#endif // ONNX_MLIR_COMPILER_MACROS_H
