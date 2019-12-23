//===--------------------------- main.cpp ---------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"

#include "src/builder/frontend_dialect_transformer.hpp"
#include "src/compiler/dialect/krnl/krnl_ops.hpp"
#include "src/compiler/dialect/onnx/onnx_ops.hpp"
#include "src/compiler/pass/passes.hpp"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

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

int main(int argc, char *argv[]) {
  mlir::registerDialect<mlir::ONNXOpsDialect>();
  mlir::registerDialect<mlir::KrnlOpsDialect>();

  llvm::cl::OptionCategory OnnfOptions("ONNF Options",
                                       "These are frontend options.");
  llvm::cl::opt<string> InputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(OnnfOptions));
  llvm::cl::HideUnrelatedOptions(OnnfOptions);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "ONNF MLIR modular optimizer driver\n");

  mlir::MLIRContext context;
  mlir::OwningModuleRef module;

  // Decide if the input file is an ONNX model or a model specified
  // in MLIR. The extension of the file is the decider.
  string extension =
      InputFilename.substr(InputFilename.find_last_of(".") + 1);
  bool onnx_model_provided = (extension == "onnx");
  bool mlir_model_provided = (extension == "mlir");

  if (onnx_model_provided) {
    ImportFrontendModelFile(InputFilename, context, module);
  } else if (mlir_model_provided) {
    LoadMLIR(InputFilename, context, module);
  } else {
    assert(false && "No ONNX or MLIR models provided!");
  }

  mlir::PassManager pm(&context);
  pm.addPass(mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLowerToKrnlPass());
  pm.addPass(mlir::createLowerKrnlPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createKrnlLowerToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.run(*module);

  // Write LLVM bitcode to disk.
  std::error_code EC;
  llvm::raw_fd_ostream moduleBitcodeStream("model.bc", EC,
                                           llvm::sys::fs::F_None);
  llvm::WriteBitcodeToFile(*mlir::translateModuleToLLVMIR(*module),
                           moduleBitcodeStream);
  moduleBitcodeStream.flush();

  return 0;
}
