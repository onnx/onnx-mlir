/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXHybridTransformPass.cpp -----------------------===//
//
// Hybrid ONNX transformation pass that combines conversion patterns for
// shape inference and canonicalization.
//
// TODO: add constant propagation and decomposition
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

struct InferShapesPattern : public OpInterfaceRewritePattern<ShapeInference> {
  using OpInterfaceRewritePattern<ShapeInference>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(
      ShapeInference shapeInfOp, PatternRewriter &rewriter) const override {
    // TODO: check if it's necessary to skip ops that satisfy
    // !returnsDynamicOrUnknownShape (see ShapeInferencePass.cpp)

    // Verify the operation before attempting to infer the shape of the
    // produced output(s).
    Optional<RegisteredOperationName> registeredInfo =
        shapeInfOp->getName().getRegisteredInfo();
    if (registeredInfo &&
        failed(registeredInfo->verifyInvariants(&*shapeInfOp)))
      return shapeInfOp.emitOpError("verification failed");

    // Infer the results shapes.
    if (failed(shapeInfOp.inferShapes([](Region &region) {})))
      return shapeInfOp.emitOpError("shape inference failed");

    return success();
  }
};

struct ReturnShapesPattern : public OpRewritePattern<ONNXReturnOp> {
  using OpRewritePattern<ONNXReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReturnOp returnOp, PatternRewriter &rewriter) const override {
    Operation *parent = returnOp->getParentOp();
    (void)parent;
    assert(parent && "every onnx.Return op has a parent");
    if (auto shapeInfOp = llvm::dyn_cast<ShapeInference>(parent)) {
      if (failed(shapeInfOp.inferShapes([](Region &region) {})))
        return shapeInfOp.emitOpError("shape inference failed");
    } else {
      llvm_unreachable("onnx.Return always has if/loop/scan parent");
    }
    return success();
  }
};

// The pass combines patterns for shape inference and other ONNX-to-ONNX
// transforms, controlled by the shapeInferenceOnly constructor argument.
//
// Shape inference is done with two patterns. One pattern is for regular ONNX
// ops from the ONNX specification, which all implement the ShapeInference
// interface. The other pattern is for ONNXReturnOp which terminates
// if/loop/scan subgraphs. The pass executes patterns top down so shape
// inference cascades through the ops from the graph's inputs to outputs, and
// recursively into subgraphs. Ops with subgraphs, namely if/loop/scan, are
// matched by the first pattern before the pass recurses into the subgraph. The
// recursive subgraph pass ends with the ONNXReturnOp whose pattern reruns shape
// inference for the parent if/loop/scan op. The effect is that the two runs of
// the parent if/loop/scan op each accomplishes one half of shape propagation to
// and from the subgraph: The first run propagates input shapes from the parent
// op to the subgraph and the second run propagates result shapes from the
// subgraph to the parent op.
struct ONNXHybridTransformPass
    : public PassWrapper<ONNXHybridTransformPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXHybridTransformPass)

  ONNXHybridTransformPass(bool shapeInferenceOnly)
      : shapeInferenceOnly(shapeInferenceOnly) {}

  StringRef getArgument() const override {
    return shapeInferenceOnly ? "shape-inference" : "onnx-hybrid-transform";
  }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet cumulativePatterns(context);
    cumulativePatterns.insert<InferShapesPattern>(context);
    cumulativePatterns.insert<ReturnShapesPattern>(context);

    if (!shapeInferenceOnly) {
      // canonicalization (copied from mlir/lib/Transforms/Canonicalizer.cpp)
      for (auto *dialect : context->getLoadedDialects())
        dialect->getCanonicalizationPatterns(cumulativePatterns);
      for (RegisteredOperationName op : context->getRegisteredOperations())
        op.getCanonicalizationPatterns(cumulativePatterns, context);

      // TODO: constant propagation, decomposition
    }

    patterns = FrozenRewritePatternSet(std::move(cumulativePatterns));
    return success();
  }

  void runOnOperation() override {
    // TODO: check if it's necessary to skip functions with names not
    // ending in "main_graph" (see ShapeInferencePass.cpp)
    func::FuncOp f = getOperation();
    Region &body = f.getBody();

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(body, patterns, config);

    Operation *returnOp = f.getBody().back().getTerminator();
    assert(returnOp && "function must return");
    FunctionType fty = f.getFunctionType();
    assert(f.getNumResults() == returnOp->getNumOperands() &&
           "returned results count much match function type");
    f.setType(fty.clone(fty.getInputs(), returnOp->getOperandTypes()));
  }

  const bool shapeInferenceOnly;
  FrozenRewritePatternSet patterns;
};

} // namespace

namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createONNXHybridTransformPass() {
  return std::make_unique<ONNXHybridTransformPass>(
      /*shapeInferenceOnly=*/false);
}

std::unique_ptr<mlir::Pass> createONNXShapeInferenceTransformPass() {
  return std::make_unique<ONNXHybridTransformPass>(/*shapeInferenceOnly=*/true);
}

} // namespace onnx_mlir