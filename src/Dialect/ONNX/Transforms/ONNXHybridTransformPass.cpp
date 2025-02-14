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
#include "src/Dialect/ONNX/Transforms/ConstProp.hpp"
#include "src/Dialect/ONNX/Transforms/ConvOpt.hpp"
#include "src/Dialect/ONNX/Transforms/Decompose.hpp"
#include "src/Dialect/ONNX/Transforms/Recompose.hpp"
#include "src/Dialect/ONNX/Transforms/ShapeInference.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

#include <iterator>

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
//
// The default values of max-num-rewrites-offset and max-num-rewrites-multiplier
// were calibrated to the model https://huggingface.co/xlnet-large-cased
// which has 1882 func ops and needs config.maxNumRewrites > 232 to converge.
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

  Option<bool> decomposition{*this, "decomposition",
      llvm::cl::desc("Enable decomposition in hybrid transform"),
      llvm::cl::init(true)};

  Option<bool> recomposition{*this, "recomposition",
      llvm::cl::desc("Enable recomposition in hybrid transform"),
      llvm::cl::init(true)};

  Option<int> maxNumRewritesOffset{*this, "max-num-rewrites-offset",
      llvm::cl::desc("Rewrites limit: -1 means no limit, otherwise "
                     "added to func #ops * max-num-rewrites-multiplier"),
      llvm::cl::init(20)};

  Option<float> maxNumRewritesMultiplier{*this, "max-num-rewrites-multiplier",
      llvm::cl::desc("Rewrites limit factor"), llvm::cl::init(0.2)};

  FrozenRewritePatternSet patterns;

  ONNXHybridTransformPass(bool enableRecomposition) {
    this->recomposition = enableRecomposition;
  }

  ONNXHybridTransformPass(const ONNXHybridTransformPass &pass)
      : patterns(pass.patterns) {
    copyOptionValuesFrom(&pass);
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

    if (decomposition) {
      getDecomposeONNXToONNXPatterns(cumulativePatterns);
    }

    if (recomposition) {
      getRecomposeONNXToONNXPatterns(cumulativePatterns);
    }

    patterns = FrozenRewritePatternSet(std::move(cumulativePatterns));
    return success();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    Region &body = f.getBody();

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    if (maxNumRewritesOffset == -1) {
      config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    } else {
      // Count the top level ops in f, i.e., excluding sub-regions.
      float numOps = std::distance(body.op_begin(), body.op_end());
      config.maxNumRewrites =
          maxNumRewritesOffset + maxNumRewritesMultiplier * numOps;
    }
    if (failed(applyPatternsAndFoldGreedily(body, patterns, config))) {
      llvm::errs() << "\nWarning: onnx-hybrid-transform didn't converge with "
                   << "max-num-rewrites-offset="
                   << maxNumRewritesOffset.getValue() << ", "
                   << "max-num-rewrites-multiplier="
                   << maxNumRewritesMultiplier.getValue() << "\n\n";
    }

    inferFunctionReturnShapes(f);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createONNXHybridTransformPass(
    bool enableRecomposition) {
  return std::make_unique<ONNXHybridTransformPass>(enableRecomposition);
}
