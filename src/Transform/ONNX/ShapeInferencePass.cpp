/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ShapeInferencePass.cpp - Shape Inference ---------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass performing propagation of array
// shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/ShapeInference.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

/*!
 *  Function pass that performs shape inference by iterating over a list of
 *  candidate operations and propagating the shape information until the list
 *  of operations is empty [credit MLIR authors].
 *
 * Shape inference proceeds recursively, starting with the entry point function
 * corresponding to the main computation graph. This is because sometimes an
 * operation is associated with a different (sub) computation graph in the forms
 * of mlir functions, and the operation's output shape and type depends on the
 * shape and type of that (sub) graph outputs. In such scenarios, operations can
 * initiate shape inference on its dependent (sub) graph, and resume inferring
 * its output shape only after shape inference completes for the associated
 * (sub) graph.
 *
 * In the absence of a main computation graph, we will treat every mlir
 * function as a main computation graph; this is mostly just for testing
 * purposes.
 */
class ShapeInferencePass
    : public PassWrapper<ShapeInferencePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  StringRef getArgument() const override { return "shape-inference"; }

  StringRef getDescription() const override {
    return "Shape inference for frontend dialects.";
  }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet cumulativePatterns(context);
    getShapeInferencePatterns(cumulativePatterns);
    patterns = FrozenRewritePatternSet(std::move(cumulativePatterns));
    return success();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    auto &body = f.getBody();
    if (enablePatternShapeInference) {
      GreedyRewriteConfig config;
      config.useTopDownTraversal = true;
      (void)applyPatternsAndFoldGreedily(body, patterns, config);
    } else {
      if (failed(runShapeInferenceOnRegion(body))) {
        signalPassFailure();
        return;
      }
    }
    inferFunctionReturnShapes(f);
  }

  FrozenRewritePatternSet patterns;

  static LogicalResult runShapeInferenceOnRegion(Region &r) {
    std::function<void(Region &)> doShapeInference = [](Region &region) {
      (void)ShapeInferencePass::runShapeInferenceOnRegion(region);
    };

    // Iterate on the operations that need shape inference i.e the operations
    // that return a dynamic shape or followed by a return op.
    for (Operation &op : r.getOps()) {
      // The shape of graph output has been imported from onnx protobuf model,
      // so the ops followed by a return op may not have dynamic shape output.
      // However, shape inference is still needed on these ops to infer optional
      // attributes.
      if (!containSubgraph(op) && !isUsedByReturnOp(op) &&
          !returnsDynamicOrUnknownShape(&op))
        continue;

      if (auto shape_op = llvm::dyn_cast<ShapeInferenceOpInterface>(op)) {
        // Verify the operation before attempting to infer the shape of the
        // produced output(s).
        std::optional<RegisteredOperationName> registeredInfo =
            op.getName().getRegisteredInfo();
        if (registeredInfo && failed(registeredInfo->verifyInvariants(&op)))
          return op.emitError("verification failed");

        // Attempt to infer the shape of the produced output(s).
        if (failed(shape_op.inferShapes(doShapeInference)))
          return op.emitError("shape inference failed");
      } else if (!llvm::dyn_cast<CallOpInterface>(op))
        return op.emitError("unable to infer shape of operation without shape "
                            "inference interface");
    }
    return success();
  }

  static bool isUsedByReturnOp(Operation &op) {
    return llvm::any_of(op.getUsers(), [](Operation *user) {
      // TODO: Only test for ONNXReturnOp once lit tests are converted
      //       to use onnx.Return.
      return isa<func::ReturnOp, ONNXReturnOp>(user);
    });
  }

  // Op needs shape inference when contains a subgraph
  // Temporary fix: only LoopOp is checked
  static bool containSubgraph(Operation &op) { return isa<ONNXLoopOp>(op); }
};
} // end anonymous namespace

/*!
 * Create a Shape Inference pass.
 */
std::unique_ptr<Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

} // namespace onnx_mlir
