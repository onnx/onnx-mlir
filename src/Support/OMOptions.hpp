/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- OMOptions.hpp ----------------------------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// Options that provide fine control on passes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/CommandLine.h"

// Declare the option categories.
extern llvm::cl::OptionCategory OMPassOptions;

// Declare options.
extern llvm::cl::opt<std::string> instrumentONNXOps;
extern llvm::cl::opt<bool> enableMemoryBundling;
extern llvm::cl::opt<int> onnxOpTransformThreshold;
extern llvm::cl::opt<bool> onnxOpTransformReport;
