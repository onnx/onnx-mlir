//===--------------------------- main.cpp ---------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "src/main_utils.hpp"

int main(int argc, char *argv[]) {
  registerDialectsForONNXMLIR();

  llvm::cl::OptionCategory OnnxMlirOptions("ONNX MLIR Options",
                                       "These are frontend options.");
  llvm::cl::opt<string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(OnnxMlirOptions));

  llvm::cl::opt<EmissionTargetType> emissionTarget(
      llvm::cl::desc("Choose target to emit:"),
      llvm::cl::values(
          clEnumVal(EmitONNXIR,
                    "Ingest ONNX and emit corresponding ONNX dialect."),
          clEnumVal(EmitMLIR,
                    "Lower model to MLIR built-in transformation dialect."),
          clEnumVal(EmitLLVMIR, "Lower model to LLVM IR (LLVM dialect)."),
          clEnumVal(EmitLLVMBC, "Lower model to LLVM IR and emit (to file) "
                                "LLVM bitcode for model.")),
      llvm::cl::init(EmitLLVMBC), llvm::cl::cat(OnnxMlirOptions));

  llvm::cl::HideUnrelatedOptions(OnnxMlirOptions);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "ONNX MLIR modular optimizer driver\n");

  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  processInputFile(inputFilename, emissionTarget, context, module);

  mlir::PassManager pm(&context);
  addONNXToMLIRPasses(pm);

  if (emissionTarget >= EmitMLIR) {
    addONNXToKRNLPasses(pm);
    addKRNLToAffinePasses(pm);
  }

  if (emissionTarget >= EmitLLVMIR)
    addKRNLToLLVMPasses(pm);

  if (mlir::failed(pm.run(*module)))
    return 4;

  if (emissionTarget == EmitLLVMBC) {
      // Write LLVM bitcode to disk.
      EmitLLVMBitCode(module);
      printf("LLVM bitcode written to ./model.bc");
  } else
    module->dump();

  return 0;
}
