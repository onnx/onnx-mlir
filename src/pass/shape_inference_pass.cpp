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

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "shape_inference_interface.hpp"
#include "src/dialect/onnx/onnx_ops.hpp"

#include "passes.hpp"

using namespace mlir;

// Include the auto-generated definitions for the shape inference interfaces.
#include "src/shape_inference.cpp.inc"

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
    llvm::SmallPtrSet<mlir::Operation *, 16> op_worklist;
    f.walk([&](mlir::Operation *op) {
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

      Operation *op = *nextop;
      op_worklist.erase(op);

      // Ask the operation to infer its output shapes.
      if (auto shape_op = dyn_cast<ShapeInference>(op)) {
        shape_op.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
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
      f.setType(FunctionType::get(
          f.getType().getInputs(),
          std::vector<Type>(results.begin(), results.end()), f.getContext()));
    }
  }

  /*!
   *  Check if the given operation has a dynamically shaped result.
   */
  static bool returnsDynamicShape(Operation *op) {
    // TODO: remove this check.
    // Temporary fix until more ops are supported.
    // All operations which do not return a ranked tensor type have dynamic
    // shaped outputs. All those operation need to implement the inferShape()
    // method.
    if (op->getName().getStringRef() != "onnx.Exp" &&
        op->getName().getStringRef() != "onnx.Tanh" &&
        op->getName().getStringRef() != "onnx.Sinh" &&
        op->getName().getStringRef() != "onnx.Cosh" &&
        op->getName().getStringRef() != "onnx.Cos" &&
        op->getName().getStringRef() != "onnx.Log" &&
        op->getName().getStringRef() != "onnx.Sigmoid" &&
        op->getName().getStringRef() != "onnx.HardSigmoid" &&
        op->getName().getStringRef() != "onnx.Elu" &&
        op->getName().getStringRef() != "onnx.Relu" &&
        op->getName().getStringRef() != "onnx.LeakyRelu" &&
        op->getName().getStringRef() != "onnx.Selu" &&
        op->getName().getStringRef() != "onnx.Reciprocal" &&
        op->getName().getStringRef() != "onnx.Softplus" &&
        op->getName().getStringRef() != "onnx.Softsign" &&
        op->getName().getStringRef() != "onnx.Sign" &&
        op->getName().getStringRef() != "onnx.Mul" &&
        op->getName().getStringRef() != "onnx.Add" &&
        op->getName().getStringRef() != "onnx.Div" &&
        op->getName().getStringRef() != "onnx.Sub" &&
        op->getName().getStringRef() != "onnx.And" &&
        op->getName().getStringRef() != "onnx.Or" &&
        op->getName().getStringRef() != "onnx.Xor" &&
        op->getName().getStringRef() != "onnx.Sum" &&
        op->getName().getStringRef() != "onnx.Max" &&
        op->getName().getStringRef() != "onnx.MaxPoolSingleOut" &&
        op->getName().getStringRef() != "onnx.Min" &&
        op->getName().getStringRef() != "onnx.Identity" &&
        op->getName().getStringRef() != "onnx.MatMul" &&
        op->getName().getStringRef() != "onnx.Gemm" &&
        op->getName().getStringRef() != "onnx.GemmNoBias" &&
        op->getName().getStringRef() != "onnx.Reshape" &&
        op->getName().getStringRef() != "onnx.Transpose" &&
        op->getName().getStringRef() != "onnx.Softmax" &&
        op->getName().getStringRef() != "onnx.Sqrt" &&
        op->getName().getStringRef() != "onnx.ConvNoBias" &&
        op->getName().getStringRef() != "onnx.Unsqueeze")
      return false;
    return llvm::any_of(op->getResultTypes(), [](Type result_type) {
      return !result_type.isa<RankedTensorType>();
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

static PassRegistration<ShapeInferencePass>
    pass("shape-inference", "Shape inference for frontend dialects.");
