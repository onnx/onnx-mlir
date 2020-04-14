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

  if (emissionTarget == EmitLLVMBC) {
    // Write LLVM bitcode to disk.
    string outputFilename =  outputBaseName + ".bc";
    EmitLLVMBitCode(module, outputFilename);
    printf("LLVM bitcode written to %s\n", outputFilename.c_str());
  } else {
    // If EmitONNXIR is the final target of the onnx-mlir invocation, the
    // output has all the embedded constant values elided. This is performed
    // by running a pass which eliminates the values of the constants.
    if (emissionTarget == EmitONNXIR) {
      // Save full version of the source code with all constant values
      // included to a file called:
      //
      // <name>.onnx.mlir
      //
      // The model without constants will be stored in a separate file:
      //
      // <name>.mlir
      //
      // If one of the two files is fed back into onnx-mlir, the .tmp file
      // the one that will be ingested by the onnx-mlir infrastructure since
      // it contains all the inlined constants.
      outputCodeWithConstants(module, outputBaseName, ".onnx.mlir");
      printf("Full ONNX IR Code written to %s\n",
          (outputBaseName + ".onnx.mlir").c_str());

      mlir::PassManager cleanSourcePM(&context);
      cleanSourcePM.addPass(mlir::createElideConstantValuePass());
      if (mlir::failed(cleanSourcePM.run(*module)))
        return 4;
      outputCodeWithConstants(module, outputBaseName, ".mlir");
      printf("Constant-free ONNX IR Code written to %s\n",
          (outputBaseName + ".mlir").c_str());
    }
  }

  return 0;
}
