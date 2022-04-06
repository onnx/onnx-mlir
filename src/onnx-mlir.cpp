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

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerUtils.hpp"

using namespace std;
using namespace onnx_mlir;

int main(int argc, char *argv[]) {
  mlir::MLIRContext context;
  registerDialects(context);

  // Register MLIR command line options.
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  // Parse options from argc/argv and default ONNX_MLIR_FLAG env var.
  llvm::cl::ParseCommandLineOptions(argc, argv,
      "ONNX-MLIR modular optimizer driver\n", nullptr,
      OnnxMlirEnvOptionName.c_str());

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;
  processInputFile(inputFilename, context, module, &errorMessage);
  if (!errorMessage.empty()) {
    printf("%s\n", errorMessage.c_str());
    return 1;
  }

  // Input file base name, replace path if required.
  // outputBaseName must specify a file, so ignore invalid values
  // such as ".", "..", "./", "/.", etc.
  bool b = false;
  if (outputBaseName == "" ||
      (b = std::regex_match(outputBaseName, std::regex("(.*/)*\\.*$")))) {
    if (b)
      printf("Invalid -o option value %s ignored.\n", outputBaseName.c_str());
    outputBaseName = inputFilename.substr(0, inputFilename.find_last_of("."));
  }

  return compileModule(module, context, outputBaseName, emissionTarget);
}
