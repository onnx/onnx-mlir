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
    // If this is the final target of the onnx-mlir invocation, the
    // output has all the embedded constant values elided. This is performed
    // by running a pass which eliminates the values of the constant and
    // replaces them with an empty list.
    if (emissionTarget == EmitONNXIR) {
      // Save full version of the source code with all constant values
      // included to file. If the model is fed back into onnx-mlir, it will
      // be this file that will be ingested by the onnx-mlir infrastructure.
      // The file will have the base name of the input file and a custom
      // extension.
      outputCodeWithConstants(module, outputBaseName, "onnx.mlir");
      pm.addPass(mlir::createElideConstantValuePass());
    }
  }

  if (emissionTarget >= EmitMLIR) {
    addONNXToKrnlPasses(pm);
    addKrnlToAffinePasses(pm);
  }

  if (emissionTarget >= EmitLLVMIR)
    addKrnlToLLVMPasses(pm);

  if (mlir::failed(pm.run(*module)))
    return 4;

  // // Output to temporary files which contain inlined constants.
  // if (emissionTarget == EmitONNXIR)
    
  // if (emissionTarget == EmitMLIR)
  //   outputCodeWithConstants(module, outputBaseName, "krnl.mlir");

  if (emissionTarget == EmitLLVMBC) {
    // Write LLVM bitcode to disk.
    string outputFilename =  outputBaseName + ".bc";
    EmitLLVMBitCode(module, outputFilename);
    printf("LLVM bitcode written to %s\n", outputFilename.c_str());
  } else {
    // Output to normal output.
    module->dump();
  }

  return 0;
}
