/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- onnx-mlir-opt.cpp - Optimization Driver ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "src/Accelerators/Accelerator.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/InitMLIRPasses.hpp"
#include "src/InitOMPasses.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

// Options for onnx-mlir-opt only.
static llvm::cl::OptionCategory OnnxMlirOptOptions(
    "ONNX-MLIR-OPT Options", "These are opt frontend options.");

static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
    llvm::cl::desc("<input file>"), llvm::cl::init("-"),
    llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<std::string> output_filename("o",
    llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"), llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool> split_input_file("split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool> verify_diagnostics("verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool> verify_passes("verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true), llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptOptions));

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
      OptimizationLevel = (onnx_mlir::OptLevel)num;
      return;
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
  // Scan Opt Level manually now as it is needed for initializing the OM
  // Passes.
  scanAndSetOptLevel(argc, argv);
  // Scan maccel manually now as it is needed for initializing the OM Passes.
  scanAndSetMAccel(argc, argv);

  // Hide unrelated options except common ones and the onnx-mlir-opt options
  // defined above.
  llvm::cl::HideUnrelatedOptions(
      {&onnx_mlir::OnnxMlirCommonOptions, &OnnxMlirOptOptions});

  mlir::DialectRegistry registry;
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::ONNXDialect>();
  registry.insert<mlir::KrnlDialect>();
  registry.insert<mlir::tosa::TosaDialect>();

  // Initialize accelerators if they exist.
  onnx_mlir::accel::initAccelerators(maccel);

  // Register dialects for accelerators.
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
    accel->registerDialects(registry);

  registerTransformsPasses();
  registerAffinePasses();
  func::registerFuncPasses();
  registerLinalgPasses();
  memref::registerMemRefPasses();
  registerSCFPasses();
  bufferization::registerBufferizationPasses();

  onnx_mlir::initOMPasses(OptimizationLevel);
  onnx_mlir::initMLIRPasses();

  // Initialize passes for accelerators.
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
    accel->initPasses(OptimizationLevel);

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

  if (!parseCustomEnvFlagsCommandLineOption(argc, argv, &llvm::errs()) ||
      !llvm::cl::ParseCommandLineOptions(argc, argv,
          "ONNX-MLIR modular optimizer driver\n", &llvm::errs(),
          customEnvFlags.c_str())) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to open file; " << error_message << "\n";
    return failed(LogicalResult::failure());
  }

  auto output = mlir::openOutputFile(output_filename, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to compile file; " << error_message << "\n";
    return failed(LogicalResult::failure());
  }

  // TODO(imaihal): Change preloadDialectsInContext to false.
  return failed(mlir::MlirOptMain(output->os(), std::move(file), passPipeline,
      registry, split_input_file, verify_diagnostics, verify_passes,
      allowUnregisteredDialects, /*preloadDialectsInContext*/ true));
}
