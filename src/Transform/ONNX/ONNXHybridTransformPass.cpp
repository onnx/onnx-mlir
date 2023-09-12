/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXHybridTransformPass.cpp -----------------------===//
//
// Hybrid ONNX transformation pass that combines conversion patterns for
// shape inference, canonicalization, constant propagation, and decomposition.
//
// Note that the decomposition patterns are applied "best effort" with a greedy
// rewrite, not a partial conversion with "legalization" to ensure that every
// decomposable op is decomposed.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/ConstProp.hpp"
#include "src/Transform/ONNX/Decompose.hpp"
#include "src/Transform/ONNX/ShapeInference.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {

// The pass combines patterns for shape inference and other ONNX-to-ONNX
// transforms.
//
// Suboptions make it possible to disable some transforms, e.g.,
// --onnx-hybrid-transform="canonicalization=false constant-propagation=false"
//
// Shape inference is done with patterns top down so shape
// inference cascades through the ops from the graph's inputs to outputs, and
// recursively into subgraphs. Ops with subgraphs, namely if/loop/scan, are
// matched by the first pattern before the pass recurses into the subgraph. The
// recursive subgraph pass ends with the ONNXYieldOp whose pattern reruns shape
// inference for the parent if/loop/scan op. The effect is that the 2 or 3 runs
// of the parent if/loop/scan op accomplish two phases of shape propagation to
// and from the subgraph(s): The first run propagates input shapes from the
// parent op to the subgraph(s) and the last run(s) propagate(s) result shapes
// from the subgraph(s) to the parent op.
struct ONNXHybridTransformPass
    : public PassWrapper<ONNXHybridTransformPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXHybridTransformPass)

  Option<bool> shapeInference{*this, "shape-inference",
      llvm::cl::desc("Enable shape inference in hybrid transform"),
      llvm::cl::init(true)};

  Option<bool> canonicalization{*this, "canonicalization",
      llvm::cl::desc("Enable canonicalization in hybrid transform"),
      llvm::cl::init(true)};

  Option<bool> constantPropagation{*this, "constant-propagation",
      llvm::cl::desc("Enable constant propagation in hybrid transform"),
      llvm::cl::init(true)};

  FrozenRewritePatternSet patterns;

  ONNXHybridTransformPass() = default;

  ONNXHybridTransformPass(const ONNXHybridTransformPass &pass)
      : patterns(pass.patterns) {
    shapeInference = pass.shapeInference;
    canonicalization = pass.canonicalization;
    constantPropagation = pass.constantPropagation;
  }

  StringRef getArgument() const override { return "onnx-hybrid-transform"; }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet cumulativePatterns(context);

    if (shapeInference) {
      getShapeInferencePatterns(cumulativePatterns);
    }

    if (canonicalization) {
      // canonicalization (copied from mlir/lib/Transforms/Canonicalizer.cpp)
      for (auto *dialect : context->getLoadedDialects())
        dialect->getCanonicalizationPatterns(cumulativePatterns);
      for (RegisteredOperationName op : context->getRegisteredOperations())
        op.getCanonicalizationPatterns(cumulativePatterns, context);
    }

    if (constantPropagation) {
      getConstPropONNXToONNXPatterns(cumulativePatterns);
    }

    getDecomposeONNXToONNXPatterns(cumulativePatterns);

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

    inferFunctionReturnShapes(f);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createONNXHybridTransformPass() {
  return std::make_unique<ONNXHybridTransformPass>();
}
