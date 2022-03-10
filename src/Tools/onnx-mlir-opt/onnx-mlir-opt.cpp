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
#include <mlir/Support/MlirOptMain.h>

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/InitMLIRPasses.hpp"
#include "src/InitOMPasses.hpp"
#include "src/Pass/Passes.hpp"

#ifdef __NNPA__
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#endif

using namespace mlir;

static llvm::cl::opt<std::string> input_filename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> output_filename("o",
    llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<bool> split_input_file("split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verify_diagnostics("verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verify_passes("verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false));

enum OptLevel { O0 = 0, O1, O2, O3 };
static llvm::cl::opt<OptLevel> OptimizationLevel(
    llvm::cl::desc("Optimization levels:"),
    llvm::cl::values(clEnumVal(O0, "Optimization level 0 (default)."),
        clEnumVal(O1, "Optimization level 1."),
        clEnumVal(O2, "Optimization level 2."),
        clEnumVal(O3, "Optimization level 3.")),
    llvm::cl::init(O0));

void scanAndSetOptLevel(int argc, char **argv) {
  // In decreasing order, so we pick the last one if there are many.
  for (int i = argc - 1; i > 0; --i) {
    std::string currStr(argv[i]);
    if (currStr.find("-O") != 0)
      continue;
    int num = atoi(&argv[i][2]); // Get the number starting 2 char down.
    // Silently ignore out of bound opt levels.
    if (num >= 0 && num <= 3) {
      OptimizationLevel = (OptLevel)num;
      return;
    }
  }
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();

  registry.insert<mlir::ONNXDialect>();
  registry.insert<mlir::KrnlOpsDialect>();
#ifdef __NNPA__
  registry.insert<onnx_mlir::zhigh::ZHighDialect>();
  registry.insert<onnx_mlir::zlow::ZLowDialect>();
#endif

  registerTransformsPasses();
  registerAffinePasses();
  registerLinalgPasses();
  memref::registerMemRefPasses();
  registerSCFPasses();
  registerStandardPasses();

  llvm::InitLLVM y(argc, argv);
  // Scan Opt Level manually now as it is needed for initializing the OM Passes.
  scanAndSetOptLevel(argc, argv);

  onnx_mlir::initOMPasses(OptimizationLevel);
  onnx_mlir::initMLIRPasses();

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
  llvm::cl::ParseCommandLineOptions(argc, argv,
      "ONNX-MLIR modular optimizer driver\n", nullptr, "ONNX_MLIR_OPT_FLAGS");

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  if (!error_message.empty()) {
    fprintf(stderr, "%s\n", error_message.c_str());
    return failed(LogicalResult::failure());
  }

  auto output = mlir::openOutputFile(output_filename, &error_message);
  if (!error_message.empty()) {
    fprintf(stderr, "%s\n", error_message.c_str());
    return failed(LogicalResult::failure());
  }

  // TODO(imaihal): Change preloadDialectsInContext to false.
  return failed(mlir::MlirOptMain(output->os(), std::move(file), passPipeline,
      registry, split_input_file, verify_diagnostics, verify_passes,
      allowUnregisteredDialects, /*preloadDialectsInContext*/ true));
}
