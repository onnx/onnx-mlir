//===--------------------------- main.cpp ---------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/file.hpp>

#include <boost/program_options.hpp>

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"

#include "src/builder/frontend_dialect_transformer.hpp"
#include "src/compiler/dialect/krnl/krnl_ops.hpp"
#include "src/compiler/dialect/onnx/onnx_ops.hpp"
#include "src/compiler/pass/passes.hpp"

#include "mlir/Analysis/Verifier.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

using namespace std;
using namespace onnf;

void LoadMLIR(string inputFilename, mlir::MLIRContext& context,
    mlir::OwningModuleRef& module) {
  // Handle '.mlir' input to the DLC compiler.
  // The mlir format indicates that one or more of the supported
  // representations are used in the file.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return;
  }
}

int main(int ac, char* av[]) {
  namespace po = boost::program_options;

  po::options_description desc("ONNF available options");
  // clang-format off
  desc.add_options()("help", "produce help message")(
      "onnx-model", po::value<string>()->required(),
        "onnx model file");
  // clang-format on

  // Handle command line argument with option names and positional
  // command line arguments.
  po::positional_options_description p;
  p.add("onnx-model", -1);
  po::variables_map vm;
  po::store(
      po::command_line_parser(ac, av).options(desc).positional(p).run(), vm);

  // TODO: allow multiple input files
  assert(vm.count("onnx-model") < 2 && "At most one input file can be provided!");

  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }

  mlir::registerDialect<mlir::ONNXOpsDialect>();
  mlir::registerDialect<mlir::KrnlOpsDialect>();

  mlir::MLIRContext context;
  mlir::OwningModuleRef module;

  // Decide if the input file is an ONNX model or a model specified
  // in MLIR. The extension of the file is the decider.
  string model_filename = vm["onnx-model"].as<string>();
  string extension =
      model_filename.substr(model_filename.find_last_of(".") + 1);
  bool onnx_model_provided = (extension == "onnx");
  bool mlir_model_provided = (extension == "mlir");

  if (onnx_model_provided) {
    ImportFrontendModelFile(model_filename, context, module);
  } else if (mlir_model_provided) {
    LoadMLIR(model_filename, context, module);
  } else {
    assert(false && "No ONNX or MLIR models provided!");
  }

  mlir::PassManager pm(&context);
  pm.addPass(mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.run(*module);

  return 0;
}
