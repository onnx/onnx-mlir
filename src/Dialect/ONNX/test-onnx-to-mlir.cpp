// test-onnx-to-mlir.cpp

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  if (argc != 2) {
    llvm::errs() << "Usage: test-onnx-to-mlir <input.onnx>\n";
    return 1;
  }

  mlir::MLIRContext context;

  // Register the ONNX dialect.
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ONNXDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // Import the ONNX model.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;
  onnx_mlir::ImportOptions options;

  int result = onnx_mlir::ImportFrontendModelFile(
      argv[1], context, module, &errorMessage, options);

  if (result != 0) {
    llvm::errs() << "Failed to import model: " << errorMessage << "\n";
    return 1;
  }

  llvm::outs() << "Successfully imported ONNX model!\n";
  module->print(llvm::outs());

  return 0;
}