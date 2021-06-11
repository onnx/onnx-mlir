/*
 * SPDX-License-Identifier: Apache-2.0
 */

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

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/FileUtilities.h"

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

enum EmissionTargetType {
  EmitONNXBasic,
  EmitONNXIR,
  EmitMLIR,
  EmitLLVMIR,
  EmitLib,
  EmitJNI,
};

enum InputIRLevelType {
  ONNXLevel,
  MLIRLevel,
  LLVMLevel,
};

void setExecPath(const char *argv0, void *fmain);

void LoadMLIR(std::string inputFilename, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module);

void compileModuleToSharedLibrary(
    const mlir::OwningModuleRef &module, std::string outputBaseName);

void compileModuleToJniJar(
    const mlir::OwningModuleRef &module, std::string outputBaseName);

void registerDialects(mlir::MLIRContext &context);

void addONNXToMLIRPasses(mlir::PassManager &pm);

void addONNXToKrnlPasses(mlir::PassManager &pm);

void addKrnlToAffinePasses(mlir::PassManager &pm);

void addKrnlToLLVMPasses(mlir::OpPassManager &pm);

void processInputFile(std::string inputFilename, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module);

InputIRLevelType determineInputIRLevel(mlir::OwningModuleRef &module);

void outputCode(
    mlir::OwningModuleRef &module, std::string filename, std::string extension);

void emitOutputFiles(std::string outputBaseName,
    EmissionTargetType emissionTarget, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module);

int compileModule(mlir::OwningModuleRef &module, mlir::MLIRContext &context,
    std::string outputBaseName, EmissionTargetType targetType);
