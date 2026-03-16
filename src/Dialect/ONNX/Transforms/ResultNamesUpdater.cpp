// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

bool isTensorName(Attribute attr) {
  if (auto strAttr = dyn_cast<StringAttr>(attr))
    return !strAttr.empty();
  else if (auto arrayAttr = dyn_cast<ArrayAttr>(attr))
    return !arrayAttr.empty();
  return false;
}

} // namespace

void ResultNamesUpdater::notifyOperationReplaced(
    Operation *op, Operation *replacement) {
  auto resultNamesArray = op->getAttrOfType<ArrayAttr>("ResultNames");
  if (!resultNamesArray)
    return;

  // Always overwrite on newly created op
  bool newOp = replacement->getUses().empty();
  // Check if ResultNames exist already
  auto replResultNames = replacement->getAttrOfType<ArrayAttr>("ResultNames");
  if (!newOp && replResultNames && llvm::all_of(replResultNames, isTensorName))
    return;

  replacement->setAttr("ResultNames", resultNamesArray);
}

void ResultNamesUpdater::notifyOperationReplaced(
    Operation *op, ValueRange replacement) {
  // If the op is replaced by a single op, simply copy the attribute
  if (Operation *replOp = replacement.front().getDefiningOp();
      replOp && llvm::all_of(replacement, [replOp](Value val) -> bool {
        return val.getDefiningOp() == replOp;
      })) {
    notifyOperationReplaced(op, replOp);
    return;
  }

  auto resultNamesArray = op->getAttrOfType<ArrayAttr>("ResultNames");
  if (!resultNamesArray)
    return;

  MLIRContext *ctx = op->getContext();
  for (auto [name, value] : llvm::zip_equal(resultNamesArray, replacement)) {
    if (auto replResult = dyn_cast<OpResult>(value)) {
      Operation *replOp = replResult.getOwner();

      // Get new or existing ResultNames
      SmallVector<Attribute> replResultNames(
          replOp->getNumResults(), StringAttr::get(ctx));
      if (auto existing = replOp->getAttrOfType<ArrayAttr>("ResultNames"))
        replResultNames = SmallVector<Attribute>(existing.getValue());

      bool newOp = replOp->getUses().empty();
      if (!newOp && isTensorName(replResultNames[replResult.getResultNumber()]))
        continue;

      // Replace the ResultName of current result
      replResultNames[replResult.getResultNumber()] = name;
      replOp->setAttr("ResultNames", ArrayAttr::get(ctx, replResultNames));
    }
  }
}

} // namespace onnx_mlir
