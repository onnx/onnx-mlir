/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- onnx-mlir-opt.cpp - Optimization Driver ---------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Main function for onnx-mlir-opt.
// Implements main for onnx-mlir-opt driver.
//
//===----------------------------------------------------------------------===//

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Bufferization/Pipelines/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Threading.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/ToolUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "RegisterPasses.hpp"
#include "src/Accelerators/Accelerator.hpp"
#include "src/Compiler/CompilerDialects.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Compiler/DisposableGarbageCollector.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Version/Version.hpp"

using namespace mlir;
using namespace onnx_mlir;

void scanAndSetOptLevel(int argc, char **argv) {
  // In decreasing order, so we pick the last one if there are many.
  for (int i = argc - 1; i > 0; --i) {
    std::string currStr(argv[i]);
    int num = -1;
    if (currStr.find("--O") == 0)
      num = atoi(&argv[i][3]); // Get the number starting 3 char down.
    else if (currStr.find("-O") == 0)
      num = atoi(&argv[i][2]); // Get the number starting 2 char down.
    // Silently ignore out of bound opt levels.
    if (num >= 0 && num <= 3) {
      OptimizationLevel = static_cast<OptLevel>(num);
      return;
    }
  }
}

void scanAndSetMCPU(int argc, char **argv) {
  // Scan for (deprecated) --mcpu and add them to the mcpu option.
  for (int i = argc - 1; i > 0; --i) {
    std::string currStr(argv[i]);
    if (currStr.find("--mcpu=") == 0) {
      std::string cpuKind(&argv[i][7]); // Get the string starting 7 chars down.
      setTargetCPU(cpuKind);
      break;
    }
    if (currStr.find("-mcpu=") == 0) {
      std::string cpuKind(&argv[i][6]); // Get the string starting 6 chars down.
      setTargetCPU(cpuKind);
      break;
    }
  }
}

void scanAndSetMArch(int argc, char **argv) {
  // Scan --march and add them to the march option.
  for (int i = argc - 1; i > 0; --i) {
    std::string currStr(argv[i]);
    if (currStr.find("--march=") == 0) {
      std::string archKind(
          &argv[i][8]); // Get the string starting 8 chars down.
      setTargetArch(archKind);
      break;
    }
    if (currStr.find("-march=") == 0) {
      std::string archKind(
          &argv[i][7]); // Get the string starting 7 chars down.
      setTargetArch(archKind);
      break;
    }
  }
}

void scanAndSetMAccel(int argc, char **argv) {
  // Scan accelerators and add them to the maccel option.
  for (int i = argc - 1; i > 0; --i) {
    std::string currStr(argv[i]);
    if (currStr.find("--maccel=") == 0) {
      std::string accelKind(
          &argv[i][9]); // Get the string starting 9 chars down.
      setTargetAccel(accelKind);
      break;
    }
    if (currStr.find("-maccel=") == 0) {
      std::string accelKind(
          &argv[i][8]); // Get the string starting 8 chars down.
      setTargetAccel(accelKind);
      break;
    }
  }
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Scan Opt Level manually now as it is needed to register passes
  // before command line options are parsed.
  scanAndSetOptLevel(argc, argv);

  // Scan CPU and Arch manually now as it is needed to register passes
  // before command line options are parsed.
  scanAndSetMCPU(argc, argv);
  scanAndSetMArch(argc, argv);

  // Scan maccel manually now as it is needed to initialize accelerators
  // before ParseCommandLineOptions() is called.
  scanAndSetMAccel(argc, argv);

  // Remove unrelated options except common ones and the onnx-mlir-opt options
  removeUnrelatedOptions({&OnnxMlirCommonOptions, &OnnxMlirOptOptions});

  DialectRegistry registry = registerDialects(maccel);
  registry.insert<tosa::TosaDialect>();

  bufferization::registerBufferizationPipelines();

  // Registered passes can be expressed as command line flags, so they must
  // must be registered before command line options are parsed.
  registerPasses(OptimizationLevel);

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();

  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  if (!parseCustomEnvFlagsCommandLineOption(argc, argv, &llvm::errs()) ||
      !llvm::cl::ParseCommandLineOptions(argc, argv,
          getVendorName() + " - A modular optimizer driver\n", &llvm::errs(),
          customEnvFlags.c_str())) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  initCompilerConfig();

  // Set up the input file.
  std::string error_message;
  auto file = openInputFile(inputFilename, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to open file; " << error_message << "\n";
    return 1;
  }

  auto output = openOutputFile(outputBaseName, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to compile file; " << error_message << "\n";
    return 1;
  }

  // Passes are configured with command line options so they must be configured
  // after command line parsing but before any passes are run.
  configurePasses();
  for (auto *accel : accel::Accelerator::getAccelerators())
    accel->configurePasses();

  std::unique_ptr<llvm::ThreadPoolInterface> threadPoolPtr = nullptr;
  auto passManagerSetupFn = [&](PassManager &pm) {
    MLIRContext *ctx = pm.getContext();
    // Set number of threads in the MLIRContext
    if (compilationNumThreads > 0)
      ctx->disableMultithreading();
    if (compilationNumThreads > 1) {
      threadPoolPtr = std::make_unique<llvm::DefaultThreadPool>(
          llvm::hardware_concurrency(compilationNumThreads));
      ctx->setThreadPool(*threadPoolPtr);
    }

    // MlirOptMain constructed ctx with our registry so we just load all our
    // already registered dialects.
    ctx->loadAllAvailableDialects();
    pm.addInstrumentation(std::make_unique<DisposableGarbageCollector>(ctx));
    auto errorHandler = [ctx](const Twine &msg) {
      emitError(UnknownLoc::get(ctx)) << msg;
      return failure();
    };
    return passPipeline.addToPipeline(pm, errorHandler);
  };

  MlirOptMainConfig config;
  config.setPassPipelineSetupFn(passManagerSetupFn)
      .splitInputFile(split_input_file ? kDefaultSplitMarker : "")
      .verifyDiagnostics(verify_diagnostics)
      .verifyPasses(verify_passes)
      .allowUnregisteredDialects(allowUnregisteredDialects)
      .emitBytecode(false)
      .useExplicitModule(false);

  if (failed(MlirOptMain(output->os(), std::move(file), registry, config)))
    return 1;

  output->keep();
  return 0;
}
