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

#include <iostream>
#include <regex>

#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Version/Version.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "onnx_mlir_main"

using namespace onnx_mlir;

int main(int argc, char *argv[]) {

  // Register MLIR command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::registerAsmPrinterCLOptions();

  llvm::cl::SetVersionPrinter(getVersionPrinter);

  // Remove unrelated options except common ones and the onnx-mlir options
  removeUnrelatedOptions({&OnnxMlirCommonOptions, &OnnxMlirOptions});

  if (!parseCustomEnvFlagsCommandLineOption(argc, argv, &llvm::errs()) ||
      !llvm::cl::ParseCommandLineOptions(argc, argv,
          getVendorName() + " - A modular optimizer driver\n", &llvm::errs(),
          customEnvFlags.c_str())) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  initCompilerConfig();

  // Special handling of outputBaseName to derive output filename.
  // outputBaseName must specify a file, so ignore invalid values
  // such as ".", "..", "./", "/.", etc.
  bool b = false;
  if (outputBaseName == "-" ||
      (b = std::regex_match(
           outputBaseName.substr(outputBaseName.find_last_of("/\\") + 1),
           std::regex("[\\.]*$")))) {
    if (b)
      llvm::errs() << "Invalid -o option value " << outputBaseName
                   << " ignored.\n";
    outputBaseName =
        (inputFilename == "-")
            ? "stdin"
            : inputFilename.substr(0, inputFilename.find_last_of("."));
  }

  // Create context after MLIRContextCLOptions are registered and parsed.
  mlir::MLIRContext context;
  mlir::registerOpenMPDialectTranslation(context);
  if (!context.isMultithreadingEnabled()) {
    assert(context.getNumThreads() == 1 && "1 thread if no multithreading");
    LLVM_DEBUG(llvm::dbgs() << "multithreading is disabled\n");
  }
  loadDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;
  int rc = processInputFile(inputFilename, context, module, &errorMessage);
  if (rc != 0) {
    if (!errorMessage.empty())
      llvm::errs() << errorMessage << "\n";
    return 1;
  }

  return compileModule(module, context, outputBaseName, emissionTarget);
}
