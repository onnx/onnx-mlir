//===--------------------------- main_utils.cpp ---------------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#include "src/MainUtils.hpp"
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

using namespace std;
using namespace onnx_mlir;

void LoadMLIR(string inputFilename, mlir::MLIRContext &context,
              mlir::OwningModuleRef &module) {
  // Handle '.mlir' input to the ONNX MLIR frontend.
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

void EmitLLVMBitCode(
    const mlir::OwningModuleRef &module, string outputFilename) {
  error_code error;
  llvm::raw_fd_ostream moduleBitcodeStream(outputFilename, error,
                                           llvm::sys::fs::F_None);
  llvm::WriteBitcodeToFile(*mlir::translateModuleToLLVMIR(*module),
                           moduleBitcodeStream);
  moduleBitcodeStream.flush();
}

void registerDialects() {
  mlir::registerDialect<mlir::AffineDialect>();
  mlir::registerDialect<mlir::LLVM::LLVMDialect>();
  mlir::registerDialect<mlir::loop::LoopOpsDialect>();
  mlir::registerDialect<mlir::StandardOpsDialect>();
  mlir::registerDialect<mlir::ONNXOpsDialect>();
  mlir::registerDialect<mlir::KrnlOpsDialect>();
}

void addONNXToMLIRPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createDecomposeONNXToONNXPass());
  pm.addPass(mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createShapeInferencePass());
  pm.addPass(mlir::createAttributePromotionPass());
}

void addONNXToKrnlPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createLowerToKrnlPass());
  // An additional pass of canonicalization is helpful because lowering
  // from ONNX dialect to Standard dialect exposes additional canonicalization
  // oppertunities.
  pm.addPass(mlir::createCanonicalizerPass());
}

void addKrnlToAffinePasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createLowerKrnlPass());
}

void addKrnlToLLVMPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createKrnlLowerToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void processInputFile(string inputFilename, EmissionTargetType emissionTarget,
	mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  // Decide if the input file is an ONNX model or a model specified
  // in MLIR. The extension of the file is the decider.
  string extension = inputFilename.substr(inputFilename.find_last_of(".") + 1);
  bool inputIsONNX = (extension == "onnx");
  bool inputIsMLIR = (extension == "mlir");
  assert(inputIsONNX != inputIsMLIR &&
         "Either ONNX model or MLIR file needs to be provided.");

  
  if (inputIsONNX) {
    ImportFrontendModelFile(inputFilename, context, module);
  } else {
    LoadMLIR(inputFilename, context, module);
  }
}

void outputCodeWithConstants(
    mlir::OwningModuleRef &module, string filename, string extension) {
  // Start a separate process to redirect the model output. I/O redirection
  // changes will not be visible to the parent process.
  if (fork() == 0) {
    const char * tempFilename = (filename + extension).c_str();
    freopen(tempFilename, "w", stderr);
    module->dump();
    fclose(stderr);
    exit(0);
  }
}

void readCodeWithConstants(
    mlir::OwningModuleRef &module, string filename, string extension) {
  // Read to a temporary file.
  // std::ofstream out("./model.");
  // std::streambuf *coutbuf = std::cout.rdbuf(); // save old buf
  // std::cout.rdbuf(out.rdbuf()); // redirect std::cout to model.temp
  // module->dump();
  // std::cout.rdbuf(coutbuf);
}