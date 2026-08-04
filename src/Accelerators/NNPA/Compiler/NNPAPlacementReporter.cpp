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

  // Helper lambda to count operations.
  auto countOp = [&](auto opType) {
    int count = 0;
    moduleOp.walk([&](decltype(opType) op) { count++; });
    return count;
  };

  // Count heavy ONNX operations that will run on CPU (alphabetical order).
  std::vector<std::pair<int, const char *>> opCounts;
  opCounts.push_back({countOp(mlir::ONNXConvOp()), "Conv"});
  opCounts.push_back({countOp(mlir::ONNXGemmOp()), "Gemm"});
  opCounts.push_back({countOp(mlir::ONNXGRUOp()), "GRU"});
  opCounts.push_back({countOp(mlir::ONNXLSTMOp()), "LSTM"});
  opCounts.push_back({countOp(mlir::ONNXMatMulOp()), "MatMul"});
  opCounts.push_back({countOp(mlir::ONNXMatMulIntegerOp()), "MatMulInteger"});
  opCounts.push_back({countOp(mlir::ONNXQLinearMatMulOp()), "QLinearMatMul"});
  opCounts.push_back({countOp(mlir::ONNXRNNOp()), "RNN"});
  opCounts.push_back({countOp(mlir::ONNXSoftmaxOp()), "Softmax"});

  // Build consolidated warning message.
  std::vector<std::string> opStrings;
  for (const auto &pair : opCounts) {
    if (pair.first > 0) {
      opStrings.push_back(std::to_string(pair.first) + " onnx." + pair.second);
    }
  }

  // Print single consolidated warning if any operations run on CPU.
  if (!opStrings.empty()) {
    llvm::outs() << "[Warning] There are ";
    for (size_t i = 0; i < opStrings.size(); ++i) {
      llvm::outs() << opStrings[i];
      if (i < opStrings.size() - 1)
        llvm::outs() << ", ";
    }
    llvm::outs()
        << " operations that run on CPU (not accelerated by NNPA). To get more "
           "information, recompile the model with the --onnx-op-stats option.\n";
  }
}

} // namespace zhigh
} // namespace onnx_mlir
