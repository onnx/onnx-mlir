/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- CompilerUtils.hpp -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/FileUtilities.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

#include "onnx-mlir/Compiler/OMCompilerTypes.h"

#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

void loadMLIR(std::string inputFilename, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module);

std::string compileModuleToObject(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string outputBaseName);
std::string compileModuleToSharedLibrary(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string outputBaseName);
void compileModuleToJniJar(const mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string outputBaseName);

void registerDialects(mlir::MLIRContext &context);

void processInputFile(std::string inputFilename, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *errorMessage);
void processInputArray(const void *onnxBuffer, int bufferSize,
    mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module);
onnx_mlir::InputIRLevelType determineInputIRLevel(
    mlir::OwningOpRef<mlir::ModuleOp> &module);

void outputCode(mlir::OwningOpRef<mlir::ModuleOp> &module, std::string filename,
    std::string extension);
void emitOutputFiles(std::string outputBaseName,
    onnx_mlir::EmissionTargetType emissionTarget, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module);
void emitOutput(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputBaseName,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType emissionTarget);

void setupModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputBaseName);
int compileModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputBaseName,
    onnx_mlir::EmissionTargetType emissionTarget);
