//===----- shape_inference_pass.cpp - Shape Inference ---------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass performing propagation of array
// shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/Pass.h"

#include "shape_inference_interface.hpp"
#include "src/compiler/dialect/onnx/onnx_ops.hpp"

#include "passes.hpp"

using namespace mlir;

// Include the auto-generated definitions for the shape inference interfaces.
#include "src/compiler/shape_inference.cpp.inc"

namespace {
/*!
 *  FunctionPass that performs shape inference by iterating over a list of
 *  candidate operations and propagating the shape information until the list
 *  of operations is empty [credit MLIR authors].
 */
class ShapeInferencePass : public mlir::FunctionPass<ShapeInferencePass> {
 public:
  void runOnFunction() override {
    auto f = getFunction();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation*, 16> op_worklist;
    f.walk([&](mlir::Operation* op) {
      if (returnsDynamicShape(op))
        op_worklist.insert(op);
    });

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!op_worklist.empty()) {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto nextop = llvm::find_if(op_worklist, returnsDynamicShape);
      if (nextop == op_worklist.end())
        break;

      Operation* op = *nextop;
      op_worklist.erase(op);

      // Ask the operation to infer its output shapes.
      if (auto shape_op = dyn_cast<ShapeInference>(op)) {
        shape_op.inferShapes();
      } else {
        op->emitError(
            "unable to infer shape of operation without shape "
            "inference interface");
        return signalPassFailure();
      }
    }

    // If the operation worklist isn't empty, this indicates a failure.
    if (!op_worklist.empty()) {
      f.emitError("Shape inference failed, ")
          << op_worklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
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
  static bool returnsDynamicShape(Operation* op) {
    // TODO: remove this check.
    // Temporary fix until more ops are supported.
    // All operations which do not return a ranked tensor type have dynamic
    // shaped outputs. All those operation need to implement the inferShape()
    // method.
    if (op->getName().getStringRef() != "onnx.Add" &&
	op->getName().getStringRef() != "onnx.Mul" &&
	op->getName().getStringRef() != "onnx.Div" &&
	op->getName().getStringRef() != "onnx.Sub" &&
	op->getName().getStringRef() != "onnx.And" &&
	op->getName().getStringRef() != "onnx.Or" &&
	op->getName().getStringRef() != "onnx.Xor" &&
        op->getName().getStringRef() != "onnx.MatMul" &&
        op->getName().getStringRef() != "onnx.Gemm" &&
        op->getName().getStringRef() != "onnx.FullGemm")
      return false;
    return llvm::any_of(op->getResultTypes(),
        [](Type result_type) { return !result_type.isa<RankedTensorType>(); });
  }
};
}  // end anonymous namespace

/*!
 * Create a Shape Inference pass.
 */
std::unique_ptr<mlir::Pass> mlir::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

static PassRegistration<ShapeInferencePass> pass(
     "shape-inference", "Shape inference for frontend dialects.");
