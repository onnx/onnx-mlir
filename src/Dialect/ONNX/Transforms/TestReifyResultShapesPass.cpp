/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- TestReifyResultShapesPass.cpp - Test reifyResultShapes on ONNX ---===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Inserts IR from ReifyRankedShapedTypeOpInterface::reifyResultShapes for ops
// that implement it. Intended for lit tests and manual pipeline inspection.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct TestONNXReifyResultShapesPass
    : public PassWrapper<TestONNXReifyResultShapesPass,
          OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestONNXReifyResultShapesPass)

  StringRef getArgument() const override {
    return "test-onnx-reify-result-shapes";
  }

  StringRef getDescription() const override {
    return "Test-only: call reifyResultShapes for ops implementing "
           "ReifyRankedShapedTypeOpInterface (e.g. onnx.Add).";
  }

  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      auto iface = dyn_cast<ReifyRankedShapedTypeOpInterface>(op);
      if (!iface)
        return;
      OpBuilder b(op);
      b.setInsertionPoint(op);
      mlir::ReifiedRankedShapedTypeDims dims;
      (void)iface.reifyResultShapes(b, dims);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTestONNXReifyResultShapesPass() {
  return std::make_unique<TestONNXReifyResultShapesPass>();
}

} // namespace onnx_mlir
