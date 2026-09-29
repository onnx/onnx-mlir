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
// LLVM-FREE FILE -- DO NOT ADD LLVM / MLIR / ONNX-MLIR COMPILER DEPENDENCES.
//
// Part of the lightweight onnx-mlir build used for the pip-installable
// packages (om_pyrt). Any LLVM/MLIR reference here will break those packages.
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_COMPILER_MACROS_H
#define ONNX_MLIR_COMPILER_MACROS_H

/// OM_EXTERNAL_VISIBILITY - classes, functions, and variables marked with this
/// keywork will be made public and visible outside of any shared library they
/// are linked in to.
#if defined(_WIN32)
#define OM_EXTERNAL_VISIBILITY __declspec(dllexport)
#else
#define OM_EXTERNAL_VISIBILITY
#endif

#endif // ONNX_MLIR_COMPILER_MACROS_H
