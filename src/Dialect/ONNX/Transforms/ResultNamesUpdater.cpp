// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <deque>
#include <memory>
#include <unordered_set>

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include "src/Dialect/ONNX/TensorName.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

using namespace mlir;

template <>
struct std::hash<Value> {
  size_t operator()(Value value) const { return hash_value(value); }
};

namespace onnx_mlir {

namespace {

void inferTensorNames(ValueRange replOperands) {
  // Collect the values that don't have TensorNames
  std::unordered_set<Value> workList;
  {
    std::deque<Value> stack(replOperands.begin(), replOperands.end());

    // Process stack entries, adding values to worklist
    while (!stack.empty()) {
      Value value = stack.front();
      stack.pop_front();
      if (!TensorName(value) && workList.insert(value).second) {
        if (Operation *defOp = value.getDefiningOp())
          for (Value operand : defOp->getOperands())
            stack.push_back(operand);
      }
    }
  }

  // Process worklist
  size_t wlen;
  do {
    wlen = workList.size();
    for (Value value : workList) {
      if (auto tname = TensorName::infer(value)) {
        workList.erase(value);
        break;
      }
    }
  } while (workList.size() > 0 && wlen > workList.size());
}

Value skipSCast(Operation *replacement) {
  if (auto scast = dyn_cast_if_present<quant::StorageCastOp>(replacement)) {
    // Second check to avoid (scast -> scast) and similar cases
    if (auto newRepl = scast.getInput().getDefiningOp();
        newRepl && !isa<quant::StorageCastOp, ONNXDequantizeLinearOp,
                       ONNXQuantizeLinearOp>(newRepl))
      return scast.getInput();
  }
  return nullptr;
}

OpResult skipSCast(OpResult value) {
  if (auto newRepl = skipSCast(value.getOwner()))
    return cast<OpResult>(newRepl);
  return value;
}

} // namespace

void ResultNamesUpdater::notifyOperationReplaced(
    Operation *op, Operation *replacement) {
  if (!op->hasAttrOfType<ArrayAttr>("ResultNames"))
    return;

  // Skip scasts
  if (auto newRepl = skipSCast(replacement))
    return notifyOperationReplaced(op, newRepl);

  // First, copy the ResultNames attribute for the last value
  auto resultNamesArray = op->getAttrOfType<ArrayAttr>("ResultNames");
  replacement->setAttr("ResultNames", resultNamesArray);

  // Infer the TensorNames for defining values
  inferTensorNames(replacement->getOperands());
}

void ResultNamesUpdater::notifyOperationReplaced(
    Operation *op, ValueRange replacement) {
  if (!op->hasAttrOfType<ArrayAttr>("ResultNames"))
    return;

  // If the op is replaced by a single op, use the simpler method
  if (Operation *replSingleOp = replacement.front().getDefiningOp();
      replSingleOp && replSingleOp->getResults() == replacement)
    return notifyOperationReplaced(op, replSingleOp);

  // First, copy the ResultNames attribute for the last value
  auto resultNamesArray = op->getAttrOfType<ArrayAttr>("ResultNames");
  MLIRContext *ctx = op->getContext();
  for (auto [name, value] : llvm::zip_equal(resultNamesArray, replacement)) {
    if (OpResult replResult = dyn_cast<OpResult>(value)) {
      replResult = skipSCast(replResult);
      Operation *replOp = replResult.getOwner();

      // Get new or existing ResultNames
      SmallVector<Attribute> replResultNames(
          replOp->getNumResults(), StringAttr::get(ctx));
      if (auto existing = replOp->getAttrOfType<ArrayAttr>("ResultNames"))
        replResultNames = SmallVector<Attribute>(existing.getValue());

      // Replace the ResultName of current result
      replResultNames[replResult.getResultNumber()] = name;
      replOp->setAttr("ResultNames", ArrayAttr::get(ctx, replResultNames));
    }
  }

  // Infer the TensorNames of defining values
  SmallVector<Value> inferenceVals;
  for (Value value : replacement) {
    if (Operation *defOp = value.getDefiningOp())
      inferenceVals.insert(
          inferenceVals.end(), defOp->operand_begin(), defOp->operand_end());
  }
  inferTensorNames(inferenceVals);
}

class InferTensorNamesPass
    : public PassWrapper<InferTensorNamesPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "onnx-infer-tensornames"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func->walk([](Operation *op) {
      for (auto result : op->getResults())
        TensorName::infer(result);
    });
  }
};

std::unique_ptr<mlir::Pass> createInferTensorNames() {
  return std::make_unique<InferTensorNamesPass>();
}

} // namespace onnx_mlir
