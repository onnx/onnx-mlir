/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ CompilerOptions.hpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding options.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Accelerators/Accelerator.hpp"
#include "llvm/Support/CommandLine.h"
#include <string>

extern const std::string OnnxMlirEnvOptionName;

namespace onnx_mlir {
extern llvm::cl::OptionCategory OnnxMlirOptions;

extern llvm::cl::opt<std::string> instrumentONNXOps;
extern llvm::cl::opt<bool> invokeOnnxVersionConverter;
extern llvm::cl::opt<bool> preserveLocations;
extern llvm::cl::opt<bool> printIR;
extern llvm::cl::opt<bool> preserveBitcode;
extern llvm::cl::opt<bool> preserveMLIR;
extern llvm::cl::opt<bool> useOnnxModelTypes;
extern llvm::cl::opt<int> repeatOnnxTransform;
extern llvm::cl::opt<std::string> shapeInformation;
extern llvm::cl::opt<onnx_mlir::OptLevel> OptimizationLevel;
extern llvm::cl::opt<std::string> mtriple;
extern llvm::cl::opt<std::string> mcpu;
extern llvm::cl::opt<std::string> march;
extern llvm::cl::list<onnx_mlir::accel::Accelerator::Kind> maccel;
extern llvm::cl::opt<bool> VerboseOutput;
extern llvm::cl::opt<std::string> Xopt;
extern llvm::cl::opt<std::string> Xllc;
extern llvm::cl::opt<std::string> mllvm;

extern llvm::cl::OptionCategory OMPassOptions;
extern llvm::cl::opt<std::string> instrumentONNXOps;
extern llvm::cl::opt<bool> enableMemoryBundling;
extern llvm::cl::opt<int> onnxOpTransformThreshold;
extern llvm::cl::opt<bool> onnxOpTransformReport;

void setTargetTriple(const std::string &triple);
std::string getTargetTripleOption();
void setTargetArch(const std::string &arch);
std::string getTargetArchOption();
void setTargetCPU(const std::string &cpu);
std::string getTargetCPUOption();
void setOptLevel(const onnx_mlir::OptLevel level);
std::string getOptimizationLevelOption();
void setXoptOption(const std::string &flag);
std::string getXoptOption();
void setXllcOption(const std::string &flag);
std::string getXllcOption();
void setLLVMOption(const std::string &flag);
std::string getLLVMOption();

// Options support for OMCompilerOptions.
using CompilerOptionList =
    llvm::SmallVector<std::pair<onnx_mlir::OptionKind, std::string>, 4>;

// Return 0 on success. These functions are not thread-safe and should be called
// by a single program thread.
int setCompilerOption(const onnx_mlir::OptionKind kind, const std::string &val);
int setCompilerOptions(const CompilerOptionList &list);

std::string getCompilerOption(const onnx_mlir::OptionKind kind);

} // namespace onnx_mlir
