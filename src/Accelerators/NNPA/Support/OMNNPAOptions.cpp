/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- OMOptions.cpp ----------------------------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// Options that provide fine control on passes.
//
//===----------------------------------------------------------------------===//

#include "OMNNPAOptions.hpp"

llvm::cl::OptionCategory OMNNPAPassOptions(
    "NNPA Pass Options", "These are options to provide fine control on passes");

llvm::cl::opt<std::string> instrumentZHighOps("instrument-zhigh-ops",
    llvm::cl::desc("Specify zhigh ops to be instrumented\n"
                   "\"NONE\" or \"\" for no instrument\n"
                   "\"ALL\" for all ops. \n"
                   "\"op1 op2 ...\" for the specified ops."),
    llvm::cl::init(""), llvm::cl::cat(OMNNPAPassOptions));
