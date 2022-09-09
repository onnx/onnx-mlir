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
#include <map>
#include <string>
#include <vector>

extern const std::string OnnxMlirEnvOptionName;

namespace onnx_mlir {
// Options for onnx-mlir only.
extern llvm::cl::OptionCategory OnnxMlirOptions;
// Common options shared between onnx-mlir and onnx-mlir-opt.
extern llvm::cl::OptionCategory OnnxMlirCommonOptions;

extern llvm::cl::opt<bool> invokeOnnxVersionConverter;
extern llvm::cl::opt<bool> preserveLocations;
extern llvm::cl::opt<bool> printIR;
extern llvm::cl::opt<bool> preserveBitcode;
extern llvm::cl::opt<bool> preserveLLVMIR;
extern llvm::cl::opt<bool> preserveMLIR;
extern llvm::cl::opt<bool> useOnnxModelTypes;
extern llvm::cl::opt<int> repeatOnnxTransform;
extern llvm::cl::opt<std::string> shapeInformation;
extern llvm::cl::opt<onnx_mlir::OptLevel> OptimizationLevel;
extern llvm::cl::opt<std::string> customEnvFlags;
extern llvm::cl::opt<std::string> mtriple;
extern llvm::cl::opt<std::string> mcpu;
extern llvm::cl::opt<std::string> march;
extern llvm::cl::list<onnx_mlir::accel::Accelerator::Kind> maccel;
extern llvm::cl::opt<bool> VerboseOutput;
extern llvm::cl::list<std::string> Xopt;
extern llvm::cl::list<std::string> Xllc;
extern llvm::cl::opt<std::string> mllvm;
extern llvm::cl::opt<bool> verifyInputTensors;

extern llvm::cl::opt<std::string> instrumentONNXOps;
extern llvm::cl::bits<InstrumentActions> instrumentControlBits;
extern llvm::cl::opt<bool> instrumentONNXSignature;
extern llvm::cl::opt<std::string> ONNXOpStats;
extern llvm::cl::opt<bool> enableMemoryBundling;
extern llvm::cl::opt<int> onnxOpTransformThreshold;
extern llvm::cl::opt<bool> onnxOpTransformReport;
extern llvm::cl::opt<bool> enableParallel;

// The customEnvFlags must be scanned before the normal options.
bool parseCustomEnvFlagsCommandLineOption(int argc, const char *const *argv,
    llvm::raw_ostream *errs = (llvm::raw_ostream *)nullptr);

void setCustomEnvVar(const std::string &envVarName);
void clearCustomEnvVar();
std::string getCustomEnvVarOption();

void setTargetTriple(const std::string &triple);
void clearTargetTriple();
std::string getTargetTripleOption();

void setTargetArch(const std::string &arch);
void clearTargetArch();
std::string getTargetArchOption();

void setTargetCPU(const std::string &cpu);
void clearTargetCPU();
std::string getTargetCPUOption();

int setTargetAccel(const std::string &str);
void setTargetAccel(const accel::Accelerator::Kind accel);
void clearTargetAccel();
std::string getTargetAccel();

void setOptLevel(const onnx_mlir::OptLevel level);
void clearOptLevel();
std::string getOptimizationLevelOption();

void setXoptOption(const std::vector<std::string> &flags);
void clearXoptOption();
std::vector<std::string> getXoptOption();

void setXllcOption(const std::vector<std::string> &flags);
void clearXllcOption();
std::vector<std::string> getXllcOption();

void setLLVMOption(const std::string &flag);
void clearLLVMOption();
std::string getLLVMOption();

// Options support for OMCompilerOptions.
using CompilerOptionList =
    llvm::SmallVector<std::pair<onnx_mlir::OptionKind, std::string>, 4>;

#define CCM_SHARED_LIB_DEPS "sharedLibDeps"
extern std::map<std::string, std::vector<std::string>> CompilerConfigMap;

// Return 0 on success. These functions are not thread-safe and should be called
// by a single program thread.
int setCompilerOption(const onnx_mlir::OptionKind kind, const std::string &val);
int setCompilerOptions(const CompilerOptionList &list);
void clearCompilerOption(const onnx_mlir::OptionKind kind);
std::string getCompilerOption(const onnx_mlir::OptionKind kind);

// The add and del functions are not thread-safe and should only be
// called from one thread.
std::vector<std::string> getCompilerConfig(std::string k);
void addCompilerConfig(std::string k, std::vector<std::string> v);
void delCompilerConfig(std::string k, std::vector<std::string> v);

} // namespace onnx_mlir