/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ onnx-mlir.cpp - Compiler Driver  ------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
// Main function for onnx-mlir.
// Implements main for onnx-mlir driver.
//===----------------------------------------------------------------------===//

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Version/Version.hpp"
#include <iostream>

using namespace std;
using namespace onnx_mlir;

extern llvm::cl::OptionCategory onnx_mlir::OnnxMlirOptions;

int main(int argc, char *argv[]) {
  mlir::MLIRContext context;
  registerDialects(context);

  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
      llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(OnnxMlirOptions));

  llvm::cl::opt<std::string> outputBaseName("o",
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
          clEnumVal(EmitMLIR,
              "Lower the input to MLIR built-in transformation dialect."),
          clEnumVal(
              EmitLLVMIR, "Lower the input to LLVM IR (LLVM MLIR dialect)."),
          clEnumVal(EmitObj, "Compile the input into a object file."),
          clEnumVal(
              EmitLib, "Compile the input into a shared library (default)."),
          clEnumVal(EmitJNI, "Compile the input into a jar file.")),
      llvm::cl::init(EmitLib), llvm::cl::cat(OnnxMlirOptions));

  // Register MLIR command line options.
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  llvm::cl::SetVersionPrinter(getVersionPrinter);

  // Parse options from argc/argv and default ONNX_MLIR_FLAG env var.
  llvm::cl::ParseCommandLineOptions(argc, argv,
      "ONNX-MLIR modular optimizer driver\n", nullptr,
      OnnxMlirEnvOptionName.c_str());

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;
  int rc = processInputFile(inputFilename, context, module, &errorMessage);
  if (rc != 0) {
    if (!errorMessage.empty())
      std::cerr << errorMessage << std::endl;
    return 1;
  }

  // Input file base name, replace path if required.
  // outputBaseName must specify a file, so ignore invalid values
  // such as ".", "..", "./", "/.", etc.
  bool b = false;
  if (outputBaseName == "" ||
      (b = std::regex_match(
           outputBaseName.substr(outputBaseName.find_last_of("/\\") + 1),
           std::regex("[\\.]*$")))) {
    if (b)
      std::cerr << "Invalid -o option value " << outputBaseName << " ignored."
                << std::endl;
    outputBaseName = inputFilename.substr(0, inputFilename.find_last_of("."));
  }

  return compileModule(module, context, outputBaseName, emissionTarget);
}
