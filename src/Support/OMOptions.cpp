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

#include "src/Support/OMOptions.hpp"

llvm::cl::OptionCategory OMPassOptions("ONNX-MLIR Pass Options",
    "These are options to provide fine control on passes");

llvm::cl::opt<std::string> instrumentONNXOps("instrument-onnx-ops",
    llvm::cl::desc("Specify onnx ops to be instrumented\n"
                   "\"NONE\" or \"\" for no instrument\n"
                   "\"ALL\" for all ops. \n"
                   "\"op1 op2 ...\" for the specified ops."),
    llvm::cl::init(""), llvm::cl::cat(OMPassOptions));

llvm::cl::opt<bool> enableMemoryBundling("enable-memory-bundling",
    llvm::cl::desc(
        "Enable memory bundling related optimizations (default=false)\n"
        "Set to 'false' if you experience significant compile time."),
    llvm::cl::init(false), llvm::cl::cat(OMPassOptions));

llvm::cl::opt<int> onnxOpTransformThreshold("onnx-op-transform-threshold",
    llvm::cl::desc(
        "Max iteration for dynamic op transform passes (default=3).\n"
        "If set to 0, onnxOpTransformPass will be disabled, and\n"
        "static iteration will be used"),
    llvm::cl::init(3), llvm::cl::cat(OMPassOptions));

llvm::cl::opt<bool> onnxOpTransformReport("onnx-op-transform-report",
    llvm::cl::desc("Report diagnostic info for op transform passes."),
    llvm::cl::init(false), llvm::cl::cat(OMPassOptions));

llvm::cl::opt<std::string> acceleratorTarget("maccel",
    llvm::cl::desc("Specify an accelerator to generate code for\n"
                   "\"NONE\" or \"\" for no accelerator\n"),
    llvm::cl::init(""), llvm::cl::cat(OMPassOptions));
