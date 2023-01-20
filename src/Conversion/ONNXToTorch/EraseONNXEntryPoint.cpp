/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- EraseModuleInitializer.cpp - ONNX dialects to Torch lowering
//---------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// ====================================================================================
//
// This file implements a pass for removing the ONNXEntryPointOp for
// compatibility when converting to Torch backend IR.
//
//===------------------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
struct EraseONNXEntryPointPass
    : public PassWrapper<EraseONNXEntryPointPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "erase-onnx-entry-point"; }

  StringRef getDescription() const override {
    return "Erase ONNXEntryPointOp.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  EraseONNXEntryPointPass() = default;
  EraseONNXEntryPointPass(const EraseONNXEntryPointPass &pass)
      : PassWrapper<EraseONNXEntryPointPass, OperationPass<ModuleOp>>() {}

  void runOnOperation() override {
    auto walkResult = getOperation().walk([](ONNXEntryPointOp op) {
      op.erase();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};
} // namespace onnx_mlir

// std::unique_ptr<OperationPass<ModuleOp>>
std::unique_ptr<mlir::Pass> onnx_mlir::createEraseONNXEntryPointPass() {
  return std::make_unique<EraseONNXEntryPointPass>();
}
