/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
extern llvm::cl::OptionCategory OnnxMlirOptions;

int main(int argc, char *argv[]) {
  setExecPath(argv[0], (void *)main);
  mlir::MLIRContext context;
  registerDialects(context);
  registerPassManagerCLOptions();

  llvm::cl::opt<string> inputFilename(llvm::cl::Positional,
      llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(OnnxMlirOptions));

  llvm::cl::opt<string> outputBaseName("o",
      llvm::cl::desc("Base path for output files, extensions will be added."),
      llvm::cl::value_desc("path"), llvm::cl::cat(OnnxMlirOptions),
      llvm::cl::ValueRequired);
  llvm::cl::opt<EmissionTargetType> emissionTarget(
      llvm::cl::desc("Choose target to emit:"),
      llvm::cl::values(
          clEnumVal(EmitONNXBasic,
              "Ingest ONNX and emit the basic ONNX operations without "
              "inferred shapes."),
          clEnumVal(
              EmitONNXIR, "Ingest ONNX and emit corresponding ONNX dialect."),
          clEnumVal(
              EmitMLIR, "Lower model to MLIR built-in transformation dialect."),
          clEnumVal(EmitApollo,
              "Lower model to Apollo: (1) emit (to file) .s for TVP. (2) emit "
              "C++ NEPAL (to file) for TCP."),
          clEnumVal(EmitLLVMIR, "Lower model to LLVM IR (LLVM dialect)."),
          clEnumVal(EmitLib, "Lower model to LLVM IR, emit (to file) "
                             "LLVM bitcode for model, compile and link it to a "
                             "shared library."),
          clEnumVal(EmitJNI, "Lower model to LLVM IR -> LLVM bitcode "
                             "-> JNI shared library -> jar")),
      llvm::cl::init(EmitLib), llvm::cl::cat(OnnxMlirOptions));

  // llvm::cl::HideUnrelatedOptions(OnnxMlirOptions);
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "ONNX MLIR modular optimizer driver\n");

  mlir::OwningModuleRef module;
  processInputFile(inputFilename, emissionTarget, context, module);

  // Input file base name, replace path if required.
  if (outputBaseName == "")
    outputBaseName = inputFilename.substr(0, inputFilename.find_last_of("."));
  if (emissionTarget == EmitApollo)
    return compileModuleApollo(module, context, outputBaseName, emissionTarget);
  else
    return compileModule(module, context, outputBaseName, emissionTarget);
}
