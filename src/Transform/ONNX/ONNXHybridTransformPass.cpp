/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXHybridTransformPass.cpp -----------------------===//
//
// Hybrid ONNX transformation pass that combines conversion patterns for
// shape inference and canonicalization and constant propagation.
//
// TODO: add decomposition
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
#include "src/Transform/ONNX/ShapeInference.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {

// Populated by configureONNXHybridTransformPass().
struct ONNXHybridTransformPassConfiguration {
  static StringSet<> disabledClasses;
};

StringSet<> ONNXHybridTransformPassConfiguration::disabledClasses;

constexpr char kShapeInference[] = "shape-inference";
constexpr char kCanonicalization[] = "canonicalization";
constexpr char kConstantPropagation[] = "constant-propagation";
bool isClass(StringRef s) {
  static StringSet<> all{
      kShapeInference, kCanonicalization, kConstantPropagation};
  return all.contains(s);
}

// The pass combines patterns for shape inference and other ONNX-to-ONNX
// transforms, controlled by the shapeInferenceOnly constructor argument.
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

  StringRef getArgument() const override { return "onnx-hybrid-transform"; }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet cumulativePatterns(context);

    if (!ONNXHybridTransformPassConfiguration::disabledClasses.contains(
            kShapeInference)) {
      getShapeInferencePatterns(cumulativePatterns);
    }

    if (!ONNXHybridTransformPassConfiguration::disabledClasses.contains(
            kCanonicalization)) {
      // canonicalization (copied from mlir/lib/Transforms/Canonicalizer.cpp)
      for (auto *dialect : context->getLoadedDialects())
        dialect->getCanonicalizationPatterns(cumulativePatterns);
      for (RegisteredOperationName op : context->getRegisteredOperations())
        op.getCanonicalizationPatterns(cumulativePatterns, context);
    }

    if (!ONNXHybridTransformPassConfiguration::disabledClasses.contains(
            kConstantPropagation)) {
      getConstPropONNXToONNXPatterns(cumulativePatterns);
    }

    // TODO: decomposition

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

  FrozenRewritePatternSet patterns;
};

} // namespace

void onnx_mlir::configureONNXHybridTransformPass(
    ArrayRef<std::string> disabledClasses) {
  assert(llvm::all_of(disabledClasses, isClass));
  ONNXHybridTransformPassConfiguration::disabledClasses.insert(
      disabledClasses.begin(), disabledClasses.end());
}

std::unique_ptr<mlir::Pass> onnx_mlir::createONNXHybridTransformPass() {
  return std::make_unique<ONNXHybridTransformPass>();
}
