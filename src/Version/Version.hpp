/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Version.hpp -------------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines several version-related utility functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/raw_ostream.h"
#include <string>

namespace onnx_mlir {
/// Retrieves the repository path from which onnx-mlir was built.
std::string getOnnxMlirRepositoryPath();

/// Retrieves the repository path from which LLVM was built.
std::string getLLVMRepositoryPath();

/// Retrieves the repository revision number (or identifier) from which
/// this onnx-mlir was built.
std::string getOnnxMlirRevision();

/// Retrieves the repository revision number (or identifier) from which
/// LLVM was built.
std::string getLLVMRevision();

/// Retrieves the full repository version that is an amalgamation of
/// the information in getOnnxMlirRepositoryPath() and  getOnnxMlirRevision().
/// And getLLVMRepositoryPath() and getLLVMRevision() if \p
/// toIncludeLLVM.
std::string getOnnxMlirFullRepositoryVersion(bool toIncludeLLVM);

/// Retrieves the full repository version that is an amalgamation of
/// the information in getLLVMRepositoryPath() and getLLVMRevision().
std::string getLLVMFullRepositoryVersion();

/// Retrieves a string representing the complete onnx-mlir version,
/// which includes the onnx-mlir version number, the repository version,
/// and the vendor tag. And LLVM repository version and vendor tag if \p
/// toIncludeLLVM.
std::string getOnnxMlirFullVersion(bool toIncludeLLVM = true);

/// Defines a version printer used to print out the version when –version is
/// given on the command line.
void getVersionPrinter(llvm::raw_ostream &os);
} // namespace onnx_mlir
