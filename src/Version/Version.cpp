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

namespace onnx_mlir {

std::string getOnnxMlirRepositoryPath() {
#if defined(ONNX_MLIR_REPOSITORY)
  return ONNX_MLIR_REPOSITORY;
#else
  return "";
#endif
}

std::string getLLVMRepositoryPath() {
#ifdef LLVM_REPOSITORY
  return LLVM_REPOSITORY;
#else
  return "";
#endif
}

std::string getOnnxMlirRevision() {
#ifdef ONNX_MLIR_REVISION
  return ONNX_MLIR_REVISION;
#else
  return "";
#endif
}

std::string getLLVMRevision() {
#ifdef LLVM_REVISION
  return LLVM_REVISION;
#else
  return "";
#endif
}

std::string getOnnxMlirFullRepositoryVersion(bool ToIncludeLLVM) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  std::string OnnxMlirPath = getOnnxMlirRepositoryPath();
  std::string OnnxMlirRevision = getOnnxMlirRevision();
  std::string LLVMPath = getLLVMRepositoryPath();
  std::string LLVMRevision = getLLVMRevision();
  if (!OnnxMlirPath.empty() && !OnnxMlirRevision.empty()) {
    os << '(' << OnnxMlirPath << ' ' << OnnxMlirRevision;
    if (ToIncludeLLVM && !LLVMPath.empty() && !LLVMRevision.empty())
      os << ", " << LLVMPath << ' ' << LLVMRevision;
    os << ')';
  }
  return buf;
}

std::string getLLVMFullRepositoryVersion() {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  std::string Path = getLLVMRepositoryPath();
  std::string Revision = getLLVMRevision();
  if (!Path.empty() && !Revision.empty())
    os << '(' << Path << ' ' << Revision << ')';
  return buf;
}

#define ONNX_MLIR_VERSION_STRING "0.3.0"

std::string getOnnxMlirFullVersion(bool ToIncludeLLVM) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
#ifdef ONNX_MLIR_VENDOR
  os << ONNX_MLIR_VENDOR;
#endif
  os << "onnx-mlir version " ONNX_MLIR_VERSION_STRING;

  std::string repo = getOnnxMlirFullRepositoryVersion(ToIncludeLLVM);
  if (!repo.empty()) {
    os << " " << repo;
  }

  return buf;
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

void getVersionPrinter(llvm::raw_ostream &os) {
  os << getOnnxMlirFullVersion(false) << "\n";
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
  os << "  LLVM version " << LLVM_PACKAGE_VERSION << ' '
     << getLLVMFullRepositoryVersion() << '\n';
}

} // namespace onnx_mlir
