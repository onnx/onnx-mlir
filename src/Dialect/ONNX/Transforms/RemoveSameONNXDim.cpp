/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==---- RemoveSameONNXDim.cpp - Remove onnx.Dim of the same dimension ----===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// When there are two onnx.Dim operations that refer to the same dynamic
// dimension. The later onnx.Dim in the IR is replaced by the first onnx.Dim.
// DimAnalysis is used to determin if two dynamic dimensions are the same or
// not.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

struct RemoveSameONNXDimPass
    : public PassWrapper<RemoveSameONNXDimPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveSameONNXDimPass)

  RemoveSameONNXDimPass() = default;
  RemoveSameONNXDimPass(const RemoveSameONNXDimPass &pass)
      : mlir::PassWrapper<RemoveSameONNXDimPass, OperationPass<ModuleOp>>() {}

  StringRef getArgument() const override { return "remove-same-onnx-dim"; }

  StringRef getDescription() const override {
    return "Remove duplicated onnx.Dim operations that refer to the same "
           "dynamic dimension. When there are two onnx.Dim operations that "
           "refer to the same dynamic dimension, the later onnx.Dim in the IR "
           "is replaced by the first onnx.Dim. DimAnalysis is used to "
           "determine if two dynamic dimensions are the same or not.";
  }

  void runOnOperation() final;
};

void RemoveSameONNXDimPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  DimAnalysis analysis(moduleOp);
  analysis.analyze();

  DimAnalysis::DimSetMapT mapping = analysis.getGroupingResult();
  llvm::SmallDenseSet<Operation *> opToRemove;
  for (auto &entry : mapping) {
    DimAnalysis::DimSetT dimSet = entry.second;
    // Find the first onnx.Dim in the group.
    Value firstONNXDimVal = nullptr;
    SmallVector<Value> dimVals;
    for (auto &ti : dimSet) {
      Value val = ti.first;
      int64_t axis = ti.second;
      // Check if val is produced by onnx.Dim?
      if (auto arg = mlir::dyn_cast<BlockArgument>(val))
        continue;
      Operation *op = val.getDefiningOp();
      auto dimOp = mlir::dyn_cast<ONNXDimOp>(op);
      if (!dimOp) {
        bool foundONNXDim = false;
        // Explore onnx.Dim from val.
        for (Operation *user : val.getUsers()) {
          if (auto dimOp = mlir::dyn_cast<ONNXDimOp>(user)) {
            Value dimVal = dimOp.getDim();
            if (dimOp.getAxis() == axis) {
              op = user;
              val = dimVal;
              foundONNXDim = true;
              break;
            }
          }
        }
        if (!foundONNXDim)
          continue;
      }
      dimVals.emplace_back(val);
      // Update the first onnx.Dim operation in the group.
      if (firstONNXDimVal == nullptr) {
        firstONNXDimVal = val;
      } else {
        auto oldOp = firstONNXDimVal.getDefiningOp();
        if (op->isBeforeInBlock(oldOp))
          firstONNXDimVal = val;
      }
    }

    // No onnx.Dim in the group. Do nothing.
    if (firstONNXDimVal == nullptr)
      continue;

    // Replace onnx.Dim ops in the group by the first onnx.Dim.
    for (Value v : dimVals) {
      if (v != firstONNXDimVal) {
        Operation *op = v.getDefiningOp();
        v.replaceAllUsesWith(firstONNXDimVal);
        opToRemove.insert(op);
      }
    }
  }
  // Remove unused onnx.Dim.
  for (Operation *op : opToRemove)
    if (op->use_empty())
      op->erase();
}

} // namespace onnx_mlir

std::unique_ptr<mlir::Pass> onnx_mlir::createRemoveSameONNXDimPass() {
  return std::make_unique<RemoveSameONNXDimPass>();
}
