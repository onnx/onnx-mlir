/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMNNPAOptions.hpp -------------------------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// NNPA options that provide fine control on passes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/CommandLine.h"

// Declare the option categories.
extern llvm::cl::OptionCategory OMNNPAPassOptions;

// Declare options.
extern llvm::cl::opt<std::string> instrumentZHighOps;
