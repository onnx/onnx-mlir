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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace {

/*!
 * This pass verify whether each ONNX op is ready for Krnl
 * The current condition is that all input tensors have to be ranked
 */

class ONNXPreKrnlVerifyPass : public mlir::PassWrapper<ONNXPreKrnlVerifyPass,
                                  OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXPreKrnlVerifyPass)

  StringRef getArgument() const override { return "onnx-pre-krnl-verify"; }

  StringRef getDescription() const override { return "Verify onnx ops."; }

  void runOnOperation() override {
    auto function = getOperation();
    auto &funcBody = function.getBody();

    // Iterate on the operations
    for (Operation &op : funcBody.getOps()) {
      if (isa<mlir::ONNXDialect>(op.getDialect())) {
        if (failed(verifyRanked(op)) || failed(verifyPadAxes(op)))
          signalPassFailure();
      }
    }
  }

private:
  static LogicalResult verifyRanked(Operation &op) {
    for (auto ty : op.getOperandTypes()) {
      if (mlir::isa<SeqType>(ty)) {
        auto seqTy = mlir::cast<SeqType>(ty);
        if (!mlir::isa<RankedTensorType>(seqTy.getElementType())) {
          op.emitError("SeqType with unranked Sequence Element");
          return failure();
        }
      } else if (!mlir::isa<RankedTensorType>(ty) && !mlir::isa<NoneType>(ty)) {
        op.emitError("not ranked");
        return failure();
      } else if (ONNXGatherNDOp gatherNDOp =
                     llvm::dyn_cast<ONNXGatherNDOp>(op)) {
        Value indices = gatherNDOp.getIndices();
        Type indicesType = indices.getType();
        ArrayRef<int64_t> indicesShape = onnx_mlir::getShape(indices.getType());
        int64_t indicesRank = onnx_mlir::getRank(indicesType);
        if (indicesShape[indicesRank - 1] == ShapedType::kDynamic) {
          op.emitError("last dim of indices in GatherND is dynamic");
          return failure();
        }
      }
    }
    return success();
  }

  // Pad axes must all be known by now: they are needed to tell which axes
  // grow, and no constant propagation happens after this point.
  static LogicalResult verifyPadAxes(Operation &op) {
    ONNXPadOp padOp = llvm::dyn_cast<ONNXPadOp>(op);
    if (!padOp || onnx_mlir::isNoneValue(padOp.getAxes()))
      return success();
    onnx_mlir::ONNXPadOpShapeHelper shapeHelper(&op, {});
    llvm::SmallVector<onnx_mlir::IndexExpr, 4> axes;
    if (failed(shapeHelper.getPadAxes(axes)))
      return op.emitError("Pad axes must have a static number of elements");
    for (uint64_t k = 0; k < axes.size(); ++k)
      if (!axes[k].isLiteral())
        return op.emitError("Pad axes must be compile-time constants after "
                            "constant propagation; axes[")
               << k << "] is not";
    return success();
  }
};
} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createONNXPreKrnlVerifyPass() {
  return std::make_unique<ONNXPreKrnlVerifyPass>();
}
