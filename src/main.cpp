//===--------------------------- main.cpp ---------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <iostream>

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"

#include "src/builder/frontend_dialect_transformer.hpp"
#include "src/dialect/krnl/krnl_ops.hpp"
#include "src/dialect/onnx/onnx_ops.hpp"
#include "src/pass/passes.hpp"

#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

void EmitLLVMBitCode(const mlir::OwningModuleRef &module);

using namespace std;
using namespace onnf;

void LoadMLIR(string inputFilename, mlir::MLIRContext &context,
              mlir::OwningModuleRef &module) {
  // Handle '.mlir' input to the ONNF frontend.
  // The mlir format indicates that one or more of the supported
  // representations are used in the file.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return;
  }
}

void EmitLLVMBitCode(const mlir::OwningModuleRef &module) {
  error_code error;
  llvm::raw_fd_ostream moduleBitcodeStream("model.bc", error,
                                           llvm::sys::fs::F_None);
  llvm::WriteBitcodeToFile(*mlir::translateModuleToLLVMIR(*module),
                           moduleBitcodeStream);
  moduleBitcodeStream.flush();
}

int main(int argc, char *argv[]) {
  mlir::registerDialect<mlir::ONNXOpsDialect>();
  mlir::registerDialect<mlir::KrnlOpsDialect>();

  llvm::cl::OptionCategory OnnfOptions("ONNF Options",
                                       "These are frontend options.");
  llvm::cl::opt<string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(OnnfOptions));

  enum EmissionTargetType {
    EmitONNXIR,
    EmitMLIR,
    EmitLLVMIR,
    EmitLLVMBC,
  };
  llvm::cl::opt<EmissionTargetType> emissionTarget(
      llvm::cl::desc("Choose target to emit:"),
      llvm::cl::values(
          clEnumVal(EmitONNXIR, "No optimizations, enable debugging"),
          clEnumVal(EmitMLIR, "Enable trivial optimizations"),
          clEnumVal(EmitLLVMIR, "Enable default optimizations"),
          clEnumVal(EmitLLVMBC, "Enable expensive optimizations")),
      llvm::cl::init(EmitLLVMBC), llvm::cl::cat(OnnfOptions));

  llvm::cl::HideUnrelatedOptions(OnnfOptions);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "ONNF MLIR modular optimizer driver\n");

  // Decide if the input file is an ONNX model or a model specified
  // in MLIR. The extension of the file is the decider.
  string extension = inputFilename.substr(inputFilename.find_last_of(".") + 1);
  bool inputIsONNX = (extension == "onnx");
  bool inputIsMLIR = (extension == "mlir");
  assert(inputIsONNX != inputIsMLIR &&
         "Either ONNX model or MLIR file needs to be provided.");

  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  if (inputIsONNX) {
    ImportFrontendModelFile(inputFilename, context, module);
  } else {
    LoadMLIR(inputFilename, context, module);
  }

  mlir::PassManager pm(&context);
  pm.addPass(mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (emissionTarget >= EmitMLIR) {
    pm.addPass(mlir::createLowerToKrnlPass());
    pm.addPass(mlir::createLowerKrnlPass());
  }

  if (emissionTarget >= EmitLLVMIR) {
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createLowerToCFGPass());
    pm.addPass(mlir::createKrnlLowerToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;

  module->dump();

  // Write LLVM bitcode to disk.
  if (emissionTarget == EmitLLVMBC)
    EmitLLVMBitCode(module);
  return 0;
}
