/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- NNPACompilerOptions.cpp --------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Compiler Options for NNPA
//
//===----------------------------------------------------------------------===//
#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"

#define DEBUG_TYPE "NNPACompilerOptions"

namespace onnx_mlir {

llvm::cl::opt<NNPAEmissionTargetType> nnpaEmissionTarget(
    llvm::cl::desc("[Optional] Choose NNPA-related target to emit "
                   "(once selected it will cancel the other targets):"),
    llvm::cl::values(
        clEnumVal(EmitZHighIR, "Lower model to ZHigh IR (ZHigh dialect)"),
        clEnumVal(EmitZLowIR, "Lower model to ZLow IR (ZLow dialect)"),
        clEnumVal(EmitZNONE, "Do not emit NNPA-related target (default)")),
    llvm::cl::init(EmitZNONE), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::list<std::string> execNodesOnCpu{"execNodesOnCpu",
    llvm::cl::desc("Comma-separated list of node names in an onnx graph. The "
                   "specified nodes are forced to run on the CPU instead of "
                   "using the zDNN. The node name is an optional attribute "
                   "in onnx graph, which is `onnx_node_name` in ONNX IR."),
    llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore,
    llvm::cl::cat(OnnxMlirOptions)};

llvm::cl::opt<bool> nnpaClipToDLFloatRange("nnpa-clip-to-dlfloat-range",
    llvm::cl::desc("Clip CPU tensors to dlfloat range before stickification to "
                   "avoid out-of-range. Only clip Softmax inputs at this "
                   "moment. Default is true."),
    llvm::cl::init(true), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> nnpaEnableZHighToOnnx("enable-zhigh-to-onnx",
    llvm::cl::desc(
        "Enabling this will convert a pattern `stick -> element-wise op -> "
        "unstick` back to an ONNX element-wise op. This conversion is called "
        "after applying all optimizations to remove stick/unstick at ZHigh "
        "level. Default is true."),
    llvm::cl::init(true), llvm::cl::cat(OnnxMlirOptions));

} // namespace onnx_mlir
