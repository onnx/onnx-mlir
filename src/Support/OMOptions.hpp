/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- main_utils.hpp ---------------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// Declare the option categories
extern llvm::cl::OptionCategory OMPassOptions;

// Declare options
extern llvm::cl::opt<std::string> instrumentONNXOps;

extern llvm::cl::opt<bool> disableMemoryBundling;

extern llvm::cl::opt<int> onnxOpTransformThreshold;
extern llvm::cl::opt<bool> onnxOpTransformReport;
