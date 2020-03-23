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
#include "mlir/InitAllDialects.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

using namespace std;
using namespace onnx_mlir;

enum EmissionTargetType {
  EmitONNXIR,
  EmitMLIR,
  EmitLLVMIR,
  EmitLLVMBC,
};

void LoadMLIR(string inputFilename, mlir::MLIRContext &context,
              mlir::OwningModuleRef &module);

void EmitLLVMBitCode(const mlir::OwningModuleRef &module);

void registerDialectsForONNXMLIR();

void addONNXToMLIRPasses(mlir::PassManager &pm);

void addONNXToKRNLPasses(mlir::PassManager &pm);

void addKRNLToAffinePasses(mlir::PassManager &pm);

void addKRNLToLLVMPasses(mlir::PassManager &pm);

void processInputFile(string inputFilename, EmissionTargetType emissionTarget,
	mlir::MLIRContext &context, mlir::OwningModuleRef &module);