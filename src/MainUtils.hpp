//===--------------------------- main_utils.hpp ---------------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <iostream>

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"

#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

enum EmissionTargetType {
  EmitONNXBasic,
  EmitONNXIR,
  EmitMLIR,
  EmitLLVMIR,
  EmitLLVMBC,
};

void LoadMLIR(std::string inputFilename, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module);

void EmitLLVMBitCode(
    const mlir::OwningModuleRef &module, std::string outputFilename);

void registerDialects();

void addONNXToMLIRPasses(mlir::PassManager &pm);

void addONNXToKrnlPasses(mlir::PassManager &pm);

void addKrnlToAffinePasses(mlir::PassManager &pm);

void addKrnlToLLVMPasses(mlir::PassManager &pm);

void processInputFile(std::string inputFilename,
    EmissionTargetType emissionTarget, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module);

void outputCode(
    mlir::OwningModuleRef &module, std::string filename, std::string extension);

void emitOutputFiles(std::string outputBaseName,
    EmissionTargetType emissionTarget, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module);