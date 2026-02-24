/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ onnx-mlir.cpp - Compiler Driver  ------------------===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
// Main function for onnx-mlir.
// Implements main for onnx-mlir driver.
//===----------------------------------------------------------------------===//

#include <filesystem>
#include <regex>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/Timing.h"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Compiler/JsonConfigObject.hpp"
#include "src/Version/Version.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "onnx_mlir_main"

using namespace onnx_mlir;

int main(int argc, char *argv[]) {
  // Register MLIR command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  llvm::cl::SetVersionPrinter(getVersionPrinter);

  // Remove unrelated options except common ones and the onnx-mlir options.
  removeUnrelatedOptions({&OnnxMlirCommonOptions, &OnnxMlirOptions});

  // STEP 1: First parse to get --config-file option.
  if (!parseCustomEnvFlagsCommandLineOption(argc, argv, &llvm::errs())) {
    llvm::errs() << "Failed to parse custom env flags\n";
    return 1;
  }

  // Temporary parse to extract config-file option.
  if (!llvm::cl::ParseCommandLineOptions(argc, argv, "", &llvm::errs(),
          nullptr, customEnvFlags.c_str())) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  // STEP 2: Load config file and extract compile options.
  std::vector<std::string> configOptions;
  std::string configPath = configFile.empty() ? "omconfig.json" : configFile;

  if (std::filesystem::exists(configPath)) {
    if (!loadCompileOptionsFromConfig(configPath, configOptions)) {
      llvm::errs() << "Warning: Failed to parse config file: " << configPath
                   << "\n";
    } else if (!configOptions.empty()) {
      llvm::outs() << "Loaded " << configOptions.size()
                   << " compile option(s) from " << configPath << "\n";
    }
  }

  // STEP 3: Merge original args with config options.
  std::vector<const char *> allArgs;
  allArgs.push_back(argv[0]); // Program name

  // Add config options first (so CLI can override).
  for (const auto &opt : configOptions) {
    allArgs.push_back(opt.c_str());
  }

  // Add original command line args (skip program name).
  for (int i = 1; i < argc; ++i) {
    allArgs.push_back(argv[i]);
  }

  // STEP 4: Final parse with merged arguments.
  if (!llvm::cl::ParseCommandLineOptions(allArgs.size(), allArgs.data(),
          getVendorName() + " - A modular optimizer driver\n", &llvm::errs(),
          nullptr, customEnvFlags.c_str())) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  initCompilerConfig();

  // Timing manager reporting enabled via "--enable-timing" compiler flag.
  timingManager.setEnabled(enableTiming);
  rootTimingScope = timingManager.getRootScope();
  auto setupTiming = rootTimingScope.nest("[onnx-mlir] Loading Dialects");

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
  // The multi-threading in MLIRContext is enabled by default. It must be
  // disabled to control the number of threads. To use single thread, simply
  // disable it. To use a specific number of threads, disable it once and then
  // set a new thread pool.
  mlir::MLIRContext context;
  std::unique_ptr<llvm::ThreadPoolInterface> threadPoolPtr = nullptr;
  if (compilationNumThreads > 0)
    context.disableMultithreading();
  if (compilationNumThreads > 1) {
    threadPoolPtr = std::make_unique<llvm::DefaultThreadPool>(
        llvm::hardware_concurrency(compilationNumThreads));
    context.setThreadPool(*threadPoolPtr);
  }

  if (!context.isMultithreadingEnabled()) {
    assert(context.getNumThreads() == 1 && "1 thread if no multithreading");
    LLVM_DEBUG(llvm::dbgs() << "multithreading is disabled\n");
  }
  loadDialects(context);
  setupTiming.stop();
  // Add the short inputFilename to the first compile phase printout so that we
  // may better determine which compilation we are dealing with.
  std::filesystem::path p(inputFilename);
  std::string modelShortName = p.filename();
  // Configure compile phase information.
  SET_TOTAL_COMPILE_PHASE(emissionTarget);
  std::string msg =
      "Importing ONNX Model to MLIR Module from \"" + modelShortName + "\"";
  showCompilePhase(msg);
  auto inputFileTiming = rootTimingScope.nest("[onnx-mlir] " + msg);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;
  int rc = processInputFile(inputFilename, context, module, &errorMessage);
  if (rc != 0) {
    if (!errorMessage.empty())
      llvm::errs() << errorMessage << "\n";
    return 1;
  }
  inputFileTiming.stop();
  return compileModule(module, context, outputBaseName, emissionTarget);
}
