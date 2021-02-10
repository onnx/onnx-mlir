//===------- ShapeInferencePass.cpp - Shape Inference ---------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass performing propagation of array
// shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"
#include "src/Interface/ShapeInferenceInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {
/*!
 *  FunctionPass that performs shape inference by iterating over a list of
 *  candidate operations and propagating the shape information until the list
 *  of operations is empty [credit MLIR authors].
 */
class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, mlir::FunctionPass> {
public:
  void runOnFunction() override {
    auto f = getFunction();

    // Iterate on the operations that need shape inference i.e the operations
    // that return a dynamic shape or followed by a return op.
    f.walk([&](mlir::Operation *op) {
      // The shape of graph output has been imported from onnx protobuf model,
      // so the ops followed by a return op may not have dynamic shape output.
      // However, shape inference is still need on these ops
      // to infer optional attributes.
      if (isUsedByReturnOp(op) || returnsDynamicShape(op)) {
        if (auto shape_op = dyn_cast<ShapeInference>(op)) {
          if (failed(shape_op.inferShapes())) {
            op->emitError("shape inference failed");
            return signalPassFailure();
          }
          // Try to fold the op if possible
          // Do nothing if fold doesn't exist because of best effor folding
          if (auto foldOp = dyn_cast<FoldInterface>(op)) {
            foldOp.tryFold();
          }
        } else {
          op->emitError("unable to infer shape of operation without shape "
                        "inference interface");
          return signalPassFailure();
        }
      }
    });

    // Populate a set to keep track of all shape fold nodes
    std::unordered_set<Operation *> shapeFoldSet;
    f.walk([&](mlir::Operation *op) {
      if (getShapeFoldingAttr(op)) {
        shapeFoldSet.insert(op);
      }
    });

    f.walk([&](mlir::Operation *op) {
      if (auto valueAttribute = getShapeFoldingAttr(op)) {
        // Only insert the constant node when the current op leads to a sink
        bool isSink = false;
        for (const auto &user : op->getUsers()) {
          auto newIt = shapeFoldSet.find(user);
          if (newIt == shapeFoldSet.end()) {
            isSink = true;
            break;
          }
        }
        if (isSink) {
          OpBuilder builder(op);
          if (!op->getResult(0)
                   .getType()
                   .dyn_cast_or_null<RankedTensorType>()) {
            op->emitError("shape folding requires type to be RankedTensorType");
            return signalPassFailure();
          }
          auto newOp = builder.create<mlir::ONNXConstantOp>(op->getLoc(),
              op->getResult(0).getType(), nullptr, valueAttribute);
          op->replaceAllUsesWith(newOp);
        } else {
          op->dropAllUses();
        }
        op->erase();
      }
    });

    int64_t dynamicOperations = 0;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op)) {
        dynamicOperations++;
      }
    });

    // If any dynamic operations remain, this indicates a failure.
    if (dynamicOperations != 0) {
      f.emitError("Shape inference failed, ")
          << dynamicOperations << " operations couldn't be inferred\n";
      return signalPassFailure();
    }

    if (auto terminator_op = f.getBody().back().getTerminator()) {
      auto results = terminator_op->getOperandTypes();
      f.setType(FunctionType::get(f.getType().getInputs(),
          std::vector<Type>(results.begin(), results.end()), f.getContext()));
    }
  }

  static bool isUsedByReturnOp(Operation *op) {
    for (auto *user : op->getUsers()) {
      if (dyn_cast<ReturnOp>(user)) {
        return true;
      }
    }
    return false;
  }

  /*!
   *  Check if the given operation has a dynamically shaped result.
   */
  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type result_type) {
      return !result_type.isa<NoneType>() &&
             !result_type.isa<RankedTensorType>();
    });
  }
};
} // end anonymous namespace

/*!
 * Create a Shape Inference pass.
 */
std::unique_ptr<mlir::Pass> mlir::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
