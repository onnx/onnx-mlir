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
// When the dynamic dimension is also a dimension of a function argument, the
// onnx.Dim operations are instead replaced by an onnx.Dim of that argument,
// placed at the beginning of the function (and created if needed). This keeps
// shape computations away from intermediate tensors, so that these tensors
// are not given extra uses only to query their shapes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
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

// Return the function whose entry block has val as argument, or null.
static func::FuncOp getFuncOfEntryArg(Value val) {
  auto arg = mlir::dyn_cast<BlockArgument>(val);
  if (!arg)
    return nullptr;
  auto funcOp = mlir::dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp());
  if (!funcOp || arg.getOwner() != &funcOp.getBody().front())
    return nullptr;
  return funcOp;
}

// Return an onnx.Dim of a function argument at the given axis, located at the
// beginning of the function entry block so that it dominates all uses in the
// function. An existing onnx.Dim in the entry block is reused (and moved),
// otherwise a new one is created.
static Value getOrCreateArgDim(BlockArgument arg, int64_t axis) {
  Block *entryBlock = arg.getOwner();
  for (Operation *user : arg.getUsers()) {
    auto dimOp = mlir::dyn_cast<ONNXDimOp>(user);
    if (dimOp && dimOp.getAxis() == axis && user->getBlock() == entryBlock) {
      user->moveBefore(entryBlock, entryBlock->begin());
      return dimOp.getDim();
    }
  }
  OpBuilder b = OpBuilder::atBlockBegin(entryBlock);
  MultiDialectBuilder<OnnxBuilder> create(b, arg.getLoc());
  return create.onnx.dim(arg, axis);
}

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
    // Function argument dimension in the group, if any. When there are many,
    // keep the one with the lowest (argument number, axis) to be
    // deterministic.
    BlockArgument funcArg = nullptr;
    int64_t funcArgAxis = -1;
    for (auto &ti : dimSet) {
      Value val = ti.first;
      int64_t axis = ti.second;
      // Record function arguments; other block arguments are skipped.
      if (auto arg = mlir::dyn_cast<BlockArgument>(val)) {
        if (getFuncOfEntryArg(arg) &&
            (!funcArg ||
                std::make_pair(arg.getArgNumber(), axis) <
                    std::make_pair(funcArg.getArgNumber(), funcArgAxis))) {
          funcArg = arg;
          funcArgAxis = axis;
        }
        continue;
      }
      // Check if val is produced by onnx.Dim?
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

    // Prefer an onnx.Dim of a function argument. Only onnx.Dim ops in the
    // same function can use it.
    func::FuncOp argFuncOp = nullptr;
    if (funcArg) {
      argFuncOp = getFuncOfEntryArg(funcArg);
      bool hasDimInFunc = llvm::any_of(dimVals, [&](Value v) {
        return v.getDefiningOp()->getParentOfType<func::FuncOp>() == argFuncOp;
      });
      if (hasDimInFunc)
        firstONNXDimVal = getOrCreateArgDim(funcArg, funcArgAxis);
      else
        argFuncOp = nullptr;
    }

    // Replace onnx.Dim ops in the group by the first onnx.Dim.
    for (Value v : dimVals) {
      if (argFuncOp &&
          v.getDefiningOp()->getParentOfType<func::FuncOp>() != argFuncOp)
        continue;
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
