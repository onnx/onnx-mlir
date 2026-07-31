/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- NNPAPlacementReporter.cpp -----------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Instrumentation pass that reports ONNX operations running on CPU.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Compiler/NNPAPlacementReporter.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace onnx_mlir {
namespace zhigh {

NNPAPlacementReporter::NNPAPlacementReporter(MLIRContext *context) {}

NNPAPlacementReporter::~NNPAPlacementReporter() {}

void NNPAPlacementReporter::runBeforePass(Pass *pass, Operation *op) {
  // Get the pass name (argument).
  llvm::StringRef passName = pass->getName();

  // Check if this is the FrontendToKrnlLoweringPass (convert-onnx-to-krnl).
  if (!passName.contains("FrontendToKrnlLoweringPass"))
    return;

  // Check if the operation is a ModuleOp.
  mlir::ModuleOp moduleOp = mlir::dyn_cast<mlir::ModuleOp>(op);
  if (!moduleOp)
    return;

  // Helper lambda to count and report operations.
  auto reportOp = [&](auto opType, const char *opName) {
    int count = 0;
    moduleOp.walk([&](decltype(opType) op) { count++; });
    if (count > 0)
      llvm::outs() << "[Warning] There are " << count << " onnx." << opName
                   << " operations that run on CPU.\n";
  };

  // Report heavy ONNX operations that will run on CPU (alphabetical order).
  reportOp(mlir::ONNXConvOp(), "Conv");
  reportOp(mlir::ONNXGemmOp(), "Gemm");
  reportOp(mlir::ONNXGRUOp(), "GRU");
  reportOp(mlir::ONNXLSTMOp(), "LSTM");
  reportOp(mlir::ONNXMatMulOp(), "MatMul");
  reportOp(mlir::ONNXMatMulIntegerOp(), "MatMulInteger");
  reportOp(mlir::ONNXQLinearMatMulOp(), "QLinearMatMul");
  reportOp(mlir::ONNXRNNOp(), "RNN");
  reportOp(mlir::ONNXSoftmaxOp(), "Softmax");
}

} // namespace zhigh
} // namespace onnx_mlir
