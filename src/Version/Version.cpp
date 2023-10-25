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
#include "llvm/TargetParser/Host.h"

namespace onnx_mlir {

std::string getVendorName() {
#if defined(ONNX_MLIR_VENDOR)
  return ONNX_MLIR_VENDOR;
#else
  return "ONNX-MLIR";
#endif
}

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

std::string getOnnxMlirCommit() {
#ifdef ONNX_MLIR_SHA
  return ONNX_MLIR_SHA;
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

std::string getOnnxMlirFullRepositoryVersion(bool toIncludeLLVM) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  std::string OnnxMlirPath = getOnnxMlirRepositoryPath();
  std::string OnnxMlirRevision = getOnnxMlirRevision();
  std::string LLVMPath = getLLVMRepositoryPath();
  std::string LLVMRevision = getLLVMRevision();
  if (!OnnxMlirPath.empty() && !OnnxMlirRevision.empty()) {
    os << '(' << OnnxMlirPath << ' ' << OnnxMlirRevision;
    if (toIncludeLLVM && !LLVMPath.empty() && !LLVMRevision.empty())
      os << ", " << LLVMPath << ' ' << LLVMRevision;
    os << ')';
  }
  return buf;
}

std::string getProductFullVersion() {
  std::string buf;
  llvm::raw_string_ostream os(buf);
#if defined(ONNX_MLIR_VENDOR)
  os << ONNX_MLIR_VENDOR << " " << PRODUCT_VERSION_MAJOR << '.';
  os << PRODUCT_VERSION_MINOR << '.';
  os << PRODUCT_VERSION_PATCH << '-' << PRODUCT_ID;
#endif
  return buf;
}

std::string getOnnxMlirCommitVersion() {
  std::string buf;
  llvm::raw_string_ostream os(buf);
#if defined(ONNX_MLIR_PRODUCT_VERSION)
  os << getProductFullVersion();
#else
  std::string onnxMlirCommit = getOnnxMlirCommit();
  os << "onnx-mlir version " ONNX_MLIR_VERSION << ' ' << '(' << onnxMlirCommit
     << ')';
#endif
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

std::string getOnnxMlirFullVersion(bool toIncludeLLVM) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  os << "onnx-mlir version " ONNX_MLIR_VERSION;
#ifdef ONNX_VERSION
  os << ", onnx version " ONNX_VERSION;
#endif
  std::string repo = getOnnxMlirFullRepositoryVersion(toIncludeLLVM);
  if (!repo.empty()) {
    os << " " << repo;
  }

  return buf;
}

#if defined(__GNUC__)
// GCC and GCC-compatible compilers define __OPTIMIZE__ when optimizations are
// enabled.
#if defined(__OPTIMIZE__)
#define ONNX_MLIR_IS_DEBUG_BUILD 0
#else
#define ONNX_MLIR_IS_DEBUG_BUILD 1
#endif
#elif defined(_MSC_VER)
// MSVC doesn't have a predefined macro indicating if optimizations are enabled.
// Use _DEBUG instead. This macro actually corresponds to the choice between
// debug and release CRTs, but it is a reasonable proxy.
#if defined(_DEBUG)
#define ONNX_MLIR_IS_DEBUG_BUILD 1
#else
#define ONNX_MLIR_IS_DEBUG_BUILD 0
#endif
#else
// Otherwise, for an unknown compiler, assume this is an optimized build.
#define ONNX_MLIR_IS_DEBUG_BUILD 0
#endif

void getVersionPrinter(llvm::raw_ostream &os) {
#if defined(ONNX_MLIR_PRODUCT_VERSION)
  os << getProductFullVersion() << "\n";
#endif
  os << getOnnxMlirFullVersion(false) << "\n";
  os << "LLVM version " << LLVM_PACKAGE_VERSION << ' '
     << getLLVMFullRepositoryVersion() << '\n';
#if ONNX_MLIR_IS_DEBUG_BUILD
  os << "DEBUG build";
#else
  os << "Optimized build";
#endif
#ifndef NDEBUG
  os << " with assertions";
#endif
  std::string CPU = std::string(llvm::sys::getHostCPUName());
  if (CPU == "generic")
    CPU = "(unknown)";
  os << ".\n";
  os << "Default target: " << llvm::sys::getDefaultTargetTriple() << '\n'
     << "Host CPU: " << CPU << '\n';
}

} // namespace onnx_mlir
