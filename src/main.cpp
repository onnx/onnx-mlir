//===--------------------------- main.cpp ---------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "src/MainUtils.hpp"

using namespace std;
using namespace onnx_mlir;

int main(int argc, char *argv[]) {
  registerDialects();

  llvm::cl::OptionCategory OnnxMlirOptions("ONNX MLIR Options",
                                       "These are frontend options.");
  llvm::cl::opt<string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(OnnxMlirOptions));

  llvm::cl::opt<EmissionTargetType> emissionTarget(
      llvm::cl::desc("Choose target to emit:"),
      llvm::cl::values(
          clEnumVal(EmitONNXBasic,
                    "Ingest ONNX and emit the basic ONNX operations without"
                    "inferred shapes."),
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

  // Input file base name.
  string outputBaseName =
      inputFilename.substr(0, inputFilename.find_last_of("."));

  mlir::PassManager pm(&context);
  if (emissionTarget >= EmitONNXIR) {
    addONNXToMLIRPasses(pm);
  }

  if (emissionTarget >= EmitMLIR) {
    addONNXToKrnlPasses(pm);
    addKrnlToAffinePasses(pm);
  }

  if (emissionTarget >= EmitLLVMIR)
    addKrnlToLLVMPasses(pm);

  if (mlir::failed(pm.run(*module)))
    return 4;

  // For EmitONNXIR and EmitMLIR the constant value are embedded in the code
  // thus making the code hard to read. These values can be elided by emitting
  // two versions of the same source code:
  // (1) a version with all the constant values included meant for being passed
  //     back to onnx-mlir for further processing and stored in:
  //
  //     <name>.onnx.mlir
  //
  // (2) a version without constants meant for being inspected by users and
  //     stored in:
  //
  //      <name>.mlir
  //
  // In the case of the LLVM Dialect IR the constant values are grouped
  // outside the function code at the beginning of the file in which case the
  // elision of these constants is not strictly required. Elision is also not
  // necessary when emitting the .bc file.
  if (emissionTarget == EmitLLVMBC) {
    // Write LLVM bitcode to disk.
    string outputFilename =  outputBaseName + ".bc";
    EmitLLVMBitCode(module, outputFilename);
    printf("LLVM bitcode written to %s\n", outputFilename.c_str());
  } else {
    // Emit the version with all constants included.
    outputCode(module, outputBaseName, ".onnx.mlir");
    printf("Full MLIR code written to %s\n",
        (outputBaseName + ".onnx.mlir").c_str());

    // Apply specific passes to clean up the code where necessary.
    mlir::PassManager cleanSourcePM(&context);
    if (emissionTarget == EmitONNXIR)
      cleanSourcePM.addPass(mlir::createElideConstantValuePass());
      // if (emissionTarget == EmitMLIR)
      //   cleanSourcePM.addPass(mlir::createElideConstGlobalValuePass());

    if (emissionTarget == EmitONNXIR || emissionTarget == EmitMLIR) {
      if (mlir::failed(cleanSourcePM.run(*module)))
        return 4;
      outputCode(module, outputBaseName, ".mlir");
      printf("Constant-free MLIR Code written to %s\n",
          (outputBaseName + ".mlir").c_str());
    }
  }

  return 0;
}
