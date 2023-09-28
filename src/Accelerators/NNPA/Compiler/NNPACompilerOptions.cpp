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

llvm::cl::opt<std::string> nnpaLoadDevicePlacementFile{
    "nnpa-load-device-placement-file",
    llvm::cl::desc(
        "Load device placement configuration from a JSON file. To "
        "have a template for the JSON file, use "
        "-save-device-placement-file=cfg.json. Note that we can use regex for "
        "string values in the JSON file to match operations."),
    llvm::cl::init(""), llvm::cl::cat(OnnxMlirOptions)};

llvm::cl::opt<std::string> nnpaSaveDevicePlacementFile{
    "nnpa-save-device-placement-file",
    llvm::cl::desc("Save device placement configuration to a JSON file."),
    llvm::cl::init(""), llvm::cl::cat(OnnxMlirOptions)};

} // namespace onnx_mlir
