// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Selective shape inference pass that runs only on standard ONNX ops,
// skipping XFE/XCompiler custom ops to avoid triggering their verification
// errors (e.g. XFEConv bias type mismatch). Only processes ops that have
// at least one unranked tensor result.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "selective-shape-inference"

using namespace mlir;

namespace onnx_mlir {

struct SelectiveShapeInferencePass
    : public PassWrapper<SelectiveShapeInferencePass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "selective-shape-inference";
  }
  StringRef getDescription() const override {
    return "Run shape inference only on standard ONNX ops with unranked "
           "results, skipping XFE/XCompiler ops";
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    bool changed = true;
    int maxIterations = 10;

    while (changed && maxIterations-- > 0) {
      changed = false;
      f.walk([&](Operation *op) {
        auto shapeInfOp = dyn_cast<ShapeInferenceOpInterface>(op);
        if (!shapeInfOp)
          return;

        StringRef name = op->getName().getStringRef();
        if (name.contains("XFE") || name.contains("XCompiler"))
          return;

        bool hasUnranked = llvm::any_of(op->getResults(), [](Value v) {
          auto tt = dyn_cast<TensorType>(v.getType());
          return tt && !tt.hasRank();
        });
        if (!hasUnranked)
          return;

        LLVM_DEBUG(llvm::dbgs()
                   << "Selective shape inference on: " << name << "\n");

        if (succeeded(shapeInfOp.inferShapes([](Region &) {})))
          changed = true;
      });
    }
  }
};

std::unique_ptr<mlir::Pass> createSelectiveShapeInferencePass() {
  return std::make_unique<SelectiveShapeInferencePass>();
}

} // namespace onnx_mlir
