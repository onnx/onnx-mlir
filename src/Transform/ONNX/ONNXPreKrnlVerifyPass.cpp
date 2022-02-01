/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ONNXPreKrnlVerifyPass.cpp - Verification -------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass that verifies whether
// ONNX ops in the function are ready for lowering to Krnl.
//
//===----------------------------------------------------------------------===//

#include <set>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/OMOptions.hpp"

using namespace mlir;

namespace {

/*!
 * This pass verify whether each ONNX op is ready for Krnl
 * The current condition is that all input tensors have to be ranked
 */

class ONNXPreKrnlVerifyPass
    : public mlir::PassWrapper<ONNXPreKrnlVerifyPass, OperationPass<FuncOp>> {

public:
  StringRef getArgument() const override { return "onnx-pre-krnl-verify"; }

  StringRef getDescription() const override { return "Verify onnx ops."; }

  void runOnOperation() override {
    auto function = getOperation();
    auto &funcBody = function.getBody();

    // Iterate on the operations
    for (Operation &op : funcBody.getOps()) {
      if (isa<mlir::ONNXOpsDialect>(op.getDialect())) {
        if (failed(verifyRanked(op)))
          signalPassFailure();
      }
    }
  }

private:
  static LogicalResult verifyRanked(Operation &op) {
    for (auto ty : op.getOperandTypes()) {
      if (!ty.isa<RankedTensorType>() && !ty.isa<NoneType>()) {
        op.emitError("not ranked");
        return failure();
      }
    }
    return success();
  }
};
} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> mlir::createONNXPreKrnlVerifyPass() {
  return std::make_unique<ONNXPreKrnlVerifyPass>();
}
