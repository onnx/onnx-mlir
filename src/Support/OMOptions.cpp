/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- MainUtils.cpp ---------------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

/*
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "ExternalUtil.hpp"
*/
#include "src/Support/OMOptions.hpp"

using namespace std;
using namespace mlir;
// using namespace onnx_mlir;

llvm::cl::OptionCategory OMPassOptions("ONNX MLIR Pass Options",
    "These are options to provide fine control on passes");

llvm::cl::opt<string> instrumentONNXOps("instrument-onnx-ops",
    llvm::cl::desc("specify onnx ops to be instrumented\n"
                   "\"NONE\" or \"\" for no instrument\n"
                   "\"ALL\" for all ops. \n"
                   "\"op1 op2 ...\" for the specified ops."),
    llvm::cl::init(""), llvm::cl::cat(OMPassOptions));

llvm::cl::opt<bool> disableMemoryBundling("disable-memory-bundling",
    llvm::cl::desc("disable memory bundling related optimizations\n"
                   "where several passes work together to bundle buffers.\n"
                   "Buffer management by MLIR is used instead.\n"
                   "Try this if you experience a significant compile time."),
    llvm::cl::init(false), llvm::cl::cat(OMPassOptions));

llvm::cl::opt<int> onnxOpTransformThreshold("onnx-op-transform-threshold",
    llvm::cl::desc("max iteration for dynamic op transform passes.\n"
                   "default value 3.\n"
                   "If set to 0, onnxOpTransformPass will be disabled, and\n"
                   "static iteration will be used"),
    llvm::cl::init(3), llvm::cl::cat(OMPassOptions));

llvm::cl::opt<bool> onnxOpTransformReport("onnx-op-transform-report",
    llvm::cl::desc(" report diagnostic info for op transform passes."),
    llvm::cl::init(false), llvm::cl::cat(OMPassOptions));
