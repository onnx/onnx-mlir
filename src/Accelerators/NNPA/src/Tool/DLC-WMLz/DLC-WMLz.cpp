//===--------------------------- main.cpp ---------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "third_party/onnx-mlir/src/Compiler/CompilerUtils.hpp"

using namespace std;
using namespace onnx_mlir;
extern llvm::cl::OptionCategory OnnxMlirOptions;

int main(int argc, char *argv[]) {
  mlir::MLIRContext context;
  registerDialects(context);

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
          clEnumVal(EmitLLVMIR, "Lower model to LLVM IR (LLVM dialect)."),
          clEnumVal(EmitObj, "Compile the input into a object file."),
          clEnumVal(EmitLib, "Lower model to LLVM IR, emit (to file) "
                             "LLVM bitcode for model, compile and link it to a "
                             "shared library."),
          clEnumVal(EmitJNI, "Lower model to LLVM IR -> LLVM bitcode "
                             "-> JNI shared library -> jar")),
      llvm::cl::init(EmitLib), llvm::cl::cat(OnnxMlirOptions));

  // llvm::cl::HideUnrelatedOptions(OnnxMlirOptions);
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "DLC ONNX Compiler\n");

  mlir::OwningModuleRef module;
  std::string errorMessage;
  processInputFile(inputFilename, context, module, &errorMessage);
  if (!errorMessage.empty()) {
    printf("%s\n", errorMessage.c_str());
    return 1;
  }

  // Input file base name, replace path if required.
  if (outputBaseName == "")
    outputBaseName = inputFilename.substr(0, inputFilename.find_last_of("."));

  return compileModule(module, context, outputBaseName, emissionTarget);
}
