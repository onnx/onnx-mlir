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

#include "src/Interface/ShapeInferenceInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {
/*!
 *  FunctionPass that performs shape inference by iterating over a list of
 *  candidate operations and propagating the shape information until the list
 *  of operations is empty [credit MLIR authors].
 */
class ShapeInferencePass : public mlir::PassWrapper<ShapeInferencePass,
                               OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    auto f = module.lookupSymbol("main_graph");
    runShapeInferenceOn(dyn_cast<mlir::FuncOp>(f));
  }

  static void runShapeInferenceOn(mlir::FuncOp f) {
    // Iterate on the operations that need shape inference i.e the operations
    // that return a dynamic shape or followed by a return op.
    f.walk([&](Operation *op) {
        std::function<void(mlir::FuncOp)> shapeInferenceFunc = &ShapeInferencePass::runShapeInferenceOn;
      // The shape of graph output has been imported from onnx protobuf model,
      // so the ops followed by a return op may not have dynamic shape output.
      // However, shape inference is still need on these ops
      // to infer optional attributes.
      if (isUsedByReturnOp(op) || returnsDynamicShape(op)) {
        if (auto shape_op = llvm::dyn_cast<ShapeInference>(op)) {
          if (failed(shape_op.inferShapes(shapeInferenceFunc))) {
            op->emitError("shape inference failed");
            return;
//            return signalPassFailure();
          }
        } else {
          op->emitError("unable to infer shape of operation without shape "
                        "inference interface");
          return;
//          return signalPassFailure();
        }
      }
    });

    int64_t dynamicOperations = 0;
    f.walk([&](Operation *op) {
      if (returnsDynamicShape(op)) {
        dynamicOperations++;
      }
    });

    // If any dynamic operations remain, this indicates a failure.
    if (dynamicOperations != 0) {
      f.emitError("Shape inference failed, ")
          << dynamicOperations << " operations couldn't be inferred\n";
      return;
//      return signalPassFailure();
    }

    if (auto terminator_op = f.getBody().back().getTerminator()) {
      auto results = terminator_op->getOperandTypes();
      f.setType(FunctionType::get(f.getType().getInputs(),
          std::vector<Type>(results.begin(), results.end()),
          f.getContext()));
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
