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
