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
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
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
#include "src/Dialect/ONNX/Transforms/LegalizeQuarkQuantizedOps.hpp"
#include "src/Dialect/ONNX/Transforms/Recompose.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Dialect/ONNX/Transforms/ShapeInference.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

#include <iterator>

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
#define GEN_PASS_DEF_ONNXHYBRIDTRANSFORMPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"
} // namespace onnx_mlir

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
    : public onnx_mlir::impl::ONNXHybridTransformPassBase<
          ONNXHybridTransformPass> {
  using Base::Base;
  FrozenRewritePatternSet patterns;

  ONNXHybridTransformPass(const ONNXHybridTransformPass &pass)
      : Base(pass), patterns(pass.patterns) {
    copyOptionValuesFrom(&pass);
  }

  // TODO: Remove once MLIR exposes a direct way to read a pass's option struct
  // back from its Pass::Option<T> members.
  [[nodiscard]] onnx_mlir::ONNXHybridTransformPassOptions getOptions() const {
    onnx_mlir::ONNXHybridTransformPassOptions options;
    options.shapeInference = shapeInference;
    options.canonicalization = canonicalization;
    options.constantPropagation = constantPropagation;
    options.qdqConstProp = qdqConstProp;
    options.decomposition = decomposition;
    options.recomposition = recomposition;
    options.quarkQuantizedOpsLegalization = quarkQuantizedOpsLegalization;
    options.maxNumRewritesOffset = maxNumRewritesOffset;
    options.maxNumRewritesMultiplier = maxNumRewritesMultiplier;
    options.enableGAPToReduceMean = enableGAPToReduceMean;
    options.enableRotaryEmbeddingRecompose = enableRotaryEmbeddingRecompose;
    options.target = target;
    options.enableConvTransposeDecompose = enableConvTransposeDecompose;
    options.enableConvTransposeDecomposeToPhasedConv =
        enableConvTransposeDecomposeToPhasedConv;
    options.enableConvTranspose1dDecomposeToPhasedConv =
        enableConvTranspose1dDecomposeToPhasedConv;
    options.enableInstanceNormDecompose = enableInstanceNormDecompose;
    options.enableGroupNormDecompose = enableGroupNormDecompose;
    options.enableMatmulNBitsDecompose = enableMatmulNBitsDecompose;
    options.enableGroupQueryAttentionDecompose =
        enableGroupQueryAttentionDecompose;
    options.enableSplitToSliceDecompose = enableSplitToSliceDecompose;
    options.enableConcatFuse = enableConcatFuse;
    options.enableLstmSeqDecompose = enableLstmSeqDecompose;
    options.enableReduceL2Decompose = enableReduceL2Decompose;
    options.enableGatherToSlice = enableGatherToSlice;
    options.enableHardSwishDecompose = enableHardSwishDecompose;
    options.enableGroupQueryAttentionCacheSlicing =
        enableGroupQueryAttentionCacheSlicing;
    return options;
  }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet cumulativePatterns(context);

    if (shapeInference) {
      getShapeInferencePatterns(cumulativePatterns);
    }

    if (quarkQuantizedOpsLegalization) {
      getLegalizeQuarkQuantizedOpsPatterns(cumulativePatterns);
      // Disable CastofConst constant propagation pattern to avoid conflicts
      // with quark quantized ops legalization which needs to consume Cast ops.
      configureConstPropONNXToONNXPass(/*roundFPToInt=*/false,
          /*expansionBound=*/-1, /*disabledPatterns=*/{"CastofConst"},
          /*constantPropIsDisabled=*/false);
    }

    if (canonicalization) {
      // canonicalization (copied from mlir/lib/Transforms/Canonicalizer.cpp)
      for (auto *dialect : context->getLoadedDialects()) {
        dialect->getCanonicalizationPatterns(cumulativePatterns);
      }
      for (RegisteredOperationName op : context->getRegisteredOperations()) {
        // Since we are manipulating ONNXCastOp's, disable any canonicalization
        // for it.
        if (quarkQuantizedOpsLegalization && op.getStringRef() == "onnx.Cast") {
          continue;
        }

        if (!enableGAPToReduceMean &&
            op.getStringRef() == "onnx.GlobalAveragePool") {
          continue;
        }
        op.getCanonicalizationPatterns(cumulativePatterns, context);
      }
    }

    if (constantPropagation) {
      getConstPropONNXToONNXPatterns(cumulativePatterns, qdqConstProp);
    }

    if (decomposition) {
      getDecomposeONNXToONNXPatterns(cumulativePatterns,
          enableConvTransposeDecompose,
          enableConvTransposeDecomposeToPhasedConv,
          enableConvTranspose1dDecomposeToPhasedConv,
          enableInstanceNormDecompose, enableGroupNormDecompose,
          enableMatmulNBitsDecompose, enableGroupQueryAttentionDecompose,
          enableSplitToSliceDecompose, enableConcatFuse, enableLstmSeqDecompose,
          enableReduceL2Decompose,
          /*disableGenericDecompositions=*/false, enableGatherToSlice,
          enableHardSwishDecompose, enableGroupQueryAttentionCacheSlicing);

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
      if (target == "stablehlo") {
        populateDecomposingONNXBeforeStablehloPatterns(
            cumulativePatterns, context);
      }
#endif
    }

    if (recomposition) {
      getRecomposeONNXToONNXPatterns(
          cumulativePatterns, enableRotaryEmbeddingRecompose);
    }

    patterns = FrozenRewritePatternSet(std::move(cumulativePatterns));
    return success();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    Region &body = f.getBody();

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    config.useTopDownTraversal = true;
    if (maxNumRewritesOffset == -1) {
      config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    } else {
      // Count all ops reachable from the function body, including ops inside
      // loop/if sub-regions.  Loop unrolling moves sub-region ops to the top
      // level, so the budget must account for them to avoid false convergence
      // failures on models with unrollable loops.
      int64_t numOps = 0;
      body.walk([&](Operation *) { ++numOps; });
      config.maxNumRewrites =
          maxNumRewritesOffset + maxNumRewritesMultiplier * numOps;
    }
    if (failed(applyPatternsGreedily(body, patterns, config))) {
      llvm::errs() << "\nWarning: onnx-hybrid-transform didn't converge with "
                   << "max-num-rewrites-offset="
                   << maxNumRewritesOffset.getValue() << ", "
                   << "max-num-rewrites-multiplier="
                   << maxNumRewritesMultiplier.getValue() << "\n\n";
    }

    if (shapeInference)
      inferFunctionReturnShapes(f);
  }
}; // namespace

} // namespace

std::optional<onnx_mlir::ONNXHybridTransformPassOptions>
onnx_mlir::parseONNXHybridTransformPassOptions(const std::string &options) {
  ONNXHybridTransformPass pass;
  if (options.empty())
    return pass.getOptions();

  std::string error;
  llvm::raw_string_ostream errorStream(error);
  auto errorHandler = [&](const Twine &message) {
    errorStream << message;
    return failure();
  };
  if (failed(pass.initializeOptions(options, errorHandler))) {
    llvm::errs() << "invalid --onnx-transform-options: " << errorStream.str();
    return std::nullopt;
  }

  return pass.getOptions();
}
