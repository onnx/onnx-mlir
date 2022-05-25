/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Version.cpp -------------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines several version-related utility functions.
//
//===----------------------------------------------------------------------===//

#include "src/Version/Version.hpp"

#include "VCSVersion.inc"
#include "llvm/Support/Host.h"

using namespace onnx_mlir;

std::string onnx_mlir::getOnnxMlirFullVersion() {
  const std::string OnnxMlirVersion = "onnx-mlir version 0.3.0";
  return
#ifdef ONNX_MLIR_VENDOR
      ONNX_MLIR_VENDOR ", " + OnnxMlirVersion;
#elif defined(ONNX_MLIR_REPOSITORY) && defined(ONNX_MLIR_REVISION) &&          \
    defined(LLVM_REPOSITORY) && defined(LLVM_REVISION)
      OnnxMlirVersion + " (" ONNX_MLIR_REPOSITORY " " ONNX_MLIR_REVISION
                        ", " LLVM_REPOSITORY " " LLVM_REVISION ")";
#else
      OnnxMlirVersion;
#endif
}

#if defined(__GNUC__)
// GCC and GCC-compatible compilers define __OPTIMIZE__ when optimizations are
// enabled.
#if defined(__OPTIMIZE__)
#define LLVM_IS_DEBUG_BUILD 0
#else
#define LLVM_IS_DEBUG_BUILD 1
#endif
#elif defined(_MSC_VER)
// MSVC doesn't have a predefined macro indicating if optimizations are enabled.
// Use _DEBUG instead. This macro actually corresponds to the choice between
// debug and release CRTs, but it is a reasonable proxy.
#if defined(_DEBUG)
#define LLVM_IS_DEBUG_BUILD 1
#else
#define LLVM_IS_DEBUG_BUILD 0
#endif
#else
// Otherwise, for an unknown compiler, assume this is an optimized build.
#define LLVM_IS_DEBUG_BUILD 0
#endif

void onnx_mlir::getVersionPrinter(llvm::raw_ostream &os) {
    os << getOnnxMlirFullVersion() << "\n";
#if LLVM_IS_DEBUG_BUILD
    os << "  DEBUG build";
#else
    os << "  Optimized build";
#endif
#ifndef NDEBUG
    os << " with assertions";
#endif
    std::string CPU = std::string(llvm::sys::getHostCPUName());
    if (CPU == "generic")
      CPU = "(unknown)";
    os << ".\n";
    os << "  Default target: " << llvm::sys::getDefaultTargetTriple() << '\n'
       << "  Host CPU: " << CPU << '\n';
    os << "  LLVM version " << LLVM_PACKAGE_VERSION << "\n";
}
