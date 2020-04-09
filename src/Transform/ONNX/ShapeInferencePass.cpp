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

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/StandardTypes.h"

#include "src/Interface/ShapeInferenceInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

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

    // Iterate on the operations that need shape inference i.e the operations
    // that return a dynamic shape.
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op)) {
        if (auto shape_op = dyn_cast<ShapeInference>(op)) {
          if (!shape_op.inferShapes()) {
            op->emitError("unable to infer shape of operation without shape "
                          "inference method");
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
        op->getName().getStringRef() != "onnx.AveragePool" &&
        op->getName().getStringRef() != "onnx.MaxPoolSingleOut" &&
        op->getName().getStringRef() != "onnx.Min" &&
        op->getName().getStringRef() != "onnx.Identity" &&
        op->getName().getStringRef() != "onnx.MatMul" &&
        op->getName().getStringRef() != "onnx.Gemm" &&
        op->getName().getStringRef() != "onnx.Reshape" &&
        op->getName().getStringRef() != "onnx.Transpose" &&
        op->getName().getStringRef() != "onnx.ReduceMax" &&
        op->getName().getStringRef() != "onnx.ReduceMin" &&
        op->getName().getStringRef() != "onnx.ReduceProd" &&
        op->getName().getStringRef() != "onnx.ReduceSum" &&
        op->getName().getStringRef() != "onnx.Softmax" &&
        op->getName().getStringRef() != "onnx.Sqrt" &&
        op->getName().getStringRef() != "onnx.Conv" &&
        op->getName().getStringRef() != "onnx.PadConstantPad" &&
        op->getName().getStringRef() != "onnx.PadConstantValuePad" &&
        op->getName().getStringRef() != "onnx.BatchNormalizationTestMode" &&
        op->getName().getStringRef() != "onnx.Abs" &&
        op->getName().getStringRef() != "onnx.Constant" &&
        op->getName().getStringRef() != "onnx.Concat" &&
        op->getName().getStringRef() != "onnx.Unsqueeze")
      return false;
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

static PassRegistration<ShapeInferencePass>
    pass("shape-inference", "Shape inference for frontend dialects.");
