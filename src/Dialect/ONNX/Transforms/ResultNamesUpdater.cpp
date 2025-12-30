// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

namespace onnx_mlir {

void ResultNamesUpdater::notifyOperationReplaced(
    mlir::Operation *op, mlir::ValueRange replacement) {
  if (!op->hasAttrOfType<mlir::ArrayAttr>("ResultNames"))
    return;

  auto resultNamesArray = op->getAttrOfType<mlir::ArrayAttr>("ResultNames");

  // If the op is replaced by a single op, simply copy the attribute
  mlir::Operation *replSingleOp = replacement.front().getDefiningOp();
  if (replSingleOp &&
      llvm::all_of(replacement, [replSingleOp](mlir::Value val) -> bool {
        return val.getDefiningOp() == replSingleOp;
      })) {
    replSingleOp->setAttr("ResultNames", resultNamesArray);
    return;
  }

  mlir::MLIRContext *ctx = op->getContext();
  auto resultNames = resultNamesArray.getAsValueRange<mlir::StringAttr>();
  for (auto [name, value] : llvm::zip_equal(resultNames, replacement)) {
    if (mlir::OpResult replResult = mlir::dyn_cast<mlir::OpResult>(value)) {
      mlir::Operation *replOp = replResult.getOwner();

      // Get new or existing ResultNames
      mlir::SmallVector<mlir::Attribute> replResultNames(
          replOp->getNumResults(), mlir::StringAttr::get(ctx, name));
      if (auto existing = replOp->getAttrOfType<mlir::ArrayAttr>("ResultNames"))
        replResultNames =
            mlir::SmallVector<mlir::Attribute>(existing.getValue());

      // Replace the ResultName of current result
      replResultNames[replResult.getResultNumber()] =
          mlir::StringAttr::get(ctx, name);
      replOp->setAttr(
          "ResultNames", mlir::ArrayAttr::get(ctx, replResultNames));
    }
  }
}

} // namespace onnx_mlir
