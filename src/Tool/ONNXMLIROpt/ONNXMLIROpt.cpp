/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- ONNXMLIROpt.cpp - Optimization Driver -----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/ToolOutputFile.h>
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

using namespace onnx_mlir;

// TODO(tjingrant): disable the following namespace import.
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
  registry.insert<mlir::tvp::TVPDialect>();
  registry.insert<mlir::ONNXOpsDialect>();
  registry.insert<mlir::KrnlOpsDialect>();

  registerTransformsPasses();
  registerAffinePasses();
  registerLinalgPasses();
  registerSCFPasses();
  registerStandardPasses();

  llvm::InitLLVM y(argc, argv);

  initOMPasses();
  initMLIRPasses();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "ONNX MLIR modular optimizer driver\n");

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  assert(file);

  auto output = mlir::openOutputFile(output_filename, &error_message);
  assert(output);

  // TODO(imaihal): Change preloadDialectsInContext to false.
  return failed(mlir::MlirOptMain(output->os(), std::move(file), passPipeline,
      registry, split_input_file, verify_diagnostics, verify_passes,
      allowUnregisteredDialects, /*preloadDialectsInContext*/ true));
}
