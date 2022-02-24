/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMDLCOptions.hpp --------------------------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// DLC options that provide fine control on passes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/CommandLine.h"

// Declare the option categories.
extern llvm::cl::OptionCategory OMDLCPassOptions;

// Declare options.
extern llvm::cl::opt<std::string> instrumentZHighOps;
