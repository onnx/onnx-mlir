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
class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, mlir::FunctionPass> {
public:
  void runOnFunction() override {
    auto f = getFunction();

    // Iterate on the operations that need shape inference i.e the operations
    // that return a dynamic shape.
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op)) {
        if (auto shape_op = dyn_cast<ShapeInference>(op)) {
          if (failed(shape_op.inferShapes())) {
            op->emitError("shape inference failed");
            return signalPassFailure();
          }
        } else {
          op->emitError("unable to infer shape of operation without shape "
                        "inference interface");
          return signalPassFailure();
        }
      }
    });

    int64_t dynamicOperations = 0;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        dynamicOperations++;
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

static PassRegistration<ShapeInferencePass> pass(
    "shape-inference", "Shape inference for frontend dialects.");
