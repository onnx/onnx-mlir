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

  Option<bool> qdqConstProp{*this, "qdq-const-prop",
      llvm::cl::desc("Enable constant propagation for QDQ"),
      llvm::cl::init(false)};

  Option<bool> decomposition{*this, "decomposition",
      llvm::cl::desc("Enable decomposition in hybrid transform"),
      llvm::cl::init(true)};

  Option<bool> recomposition{*this, "recomposition",
      llvm::cl::desc("Enable recomposition in hybrid transform"),
      llvm::cl::init(true)};

  Option<bool> quarkQuantizedOpsLegalization{*this,
      "quark-quantized-ops-legalization",
      llvm::cl::desc(
          "Enable legalization quark-quantized operations from F32 -> BF16"),
      llvm::cl::init(false)};

  Option<int> maxNumRewritesOffset{*this, "max-num-rewrites-offset",
      llvm::cl::desc("Rewrites limit: -1 means no limit, otherwise "
                     "added to func #ops * max-num-rewrites-multiplier"),
      llvm::cl::init(20)};

  Option<float> maxNumRewritesMultiplier{*this, "max-num-rewrites-multiplier",
      llvm::cl::desc("Rewrites limit factor"), llvm::cl::init(0.2)};

  Option<bool> enableConvTransposeDecompose{*this, "enable-convtranspose",
      llvm::cl::desc("Enable decomposition of ConvTranspose"),
      ::llvm::cl::init(false)};

  Option<bool> enableConvTransposeDecomposeToPhasedConv{*this,
      "enable-convtranspose-phased",
      llvm::cl::desc("Enable decomposition of ONNX ConvTranspose operator to 4 "
                     "phased Conv"),
      ::llvm::cl::init(false)};

  Option<bool> enableConvTranspose1dDecomposeToPhasedConv{*this,
      "enable-convtranspose-1d-phased",
      llvm::cl::desc(
          "Enable decomposition of ONNX ConvTranspose 1D operator to "
          "phased Conv"),
      ::llvm::cl::init(false)};

  Option<bool> enableReduceL2Decompose{*this, "enable-reducel2-decompose",
      llvm::cl::desc("Enable decomposition of ReduceL2 to "
                     "Sqrt(ReduceSumSquare(x))"),
      ::llvm::cl::init(true)};

  Option<bool> enableInstanceNormDecompose{*this,
      "enable-instancenorm-decompose",
      llvm::cl::desc("Enable decomposition of InstanceNormalization to "
                     "LayerNormalization"),
      ::llvm::cl::init(true)};

  Option<bool> enableGroupNormDecompose{*this, "enable-groupnorm-decompose",
      llvm::cl::desc("Enable decomposition of GroupNormalization to "
                     "LayerNormalization"),
      ::llvm::cl::init(true)};

  Option<bool> enableMatmulNBitsDecompose{*this, "enable-matmulnbits-decompose",
      llvm::cl::desc("Enable decomposition of Microsoft MatmulNBits to "
                     "dequantize linear and matmul ops"),
      ::llvm::cl::init(false)};

  Option<bool> enableGroupQueryAttentionDecompose{*this,
      "enable-groupqueryattention-decompose",
      llvm::cl::desc("Enable decomposition of Microsoft GroupQueryAttention to "
                     "onnx.Attention and onnx.RotaryEmbedding ops"),
      ::llvm::cl::init(true)};

  Option<bool> enableSplitToSliceDecompose{*this,
      "enable-split-to-slice-decompose",
      llvm::cl::desc("Enable decomposition of Split to Slice"),
      ::llvm::cl::init(false)};

  Option<bool> enableConcatFuse{*this, "enable-concat-fuse",
      llvm::cl::desc("Enable ConcatFusePattern in decomposition pass"),
      ::llvm::cl::init(true)};

  Option<bool> enableGAPToReduceMean{*this,
      "enable-globalaveragepool-to-reducemean",
      llvm::cl::desc(
          "Enable canonicalize from GlobalAveragePool to ReduceMean"),
      ::llvm::cl::init(true)};

  Option<bool> enableLstmSeqDecompose{*this, "enable-lstm-seq-decomposition",
      llvm::cl::desc("Enable sequence-length decomposition of LSTM (unroll a "
                     "seq_len>1 LSTM into a chain of seq_len=1 LSTMs)"),
      ::llvm::cl::init(false)};

  Option<bool> enableGatherToSlice{*this, "enable-gather-to-slice",
      llvm::cl::desc(
          "Enable decomposition of Gather with scalar index to Slice+Reshape"),
      ::llvm::cl::init(true)};

  Option<bool> enableRotaryEmbeddingRecompose{*this,
      "enable-rotary-embedding-recompose",
      llvm::cl::desc("Recompose LlamaRotaryEmbedding style RoPE "
                     "into onnx.RotaryEmbedding"),
      ::llvm::cl::init(false)};

  Option<bool> enableHardSwishDecompose{*this, "enable-hardswish-decompose",
      llvm::cl::desc("Enable decomposition of HardSwish into "
                     "x * HardSigmoid(x) (alpha=1/6, beta=0.5)"),
      ::llvm::cl::init(true)};

  FrozenRewritePatternSet patterns;

  ONNXHybridTransformPass(bool enableRecomposition,
      bool enableQuarkQuantizedOpsLegalization,
      bool enableConvTransposeDecompose,
      bool enableConvTransposeDecomposeToPhasedConv,
      bool enableConvTranspose1dDecomposeToPhasedConv,
      bool enableInstanceNormDecompose, bool enableGroupNormDecompose,
      bool enableMatmulNBitsDecompose, bool enableGroupQueryAttentionDecompose,
      bool enableSplitToSliceDecompose, bool enableConcatFuse,
      bool enableGAPToReduceMean, bool enableLstmSeqDecompose = false,
      bool enableGatherToSlice = true, bool enableReduceL2Decompose = true,
      bool enableRotaryEmbeddingRecompose = false,
      bool enableQDQConstProp = false, bool enableHardSwishDecompose = true) {
    this->recomposition = enableRecomposition;
    this->quarkQuantizedOpsLegalization = enableQuarkQuantizedOpsLegalization;
    this->enableConvTransposeDecompose = enableConvTransposeDecompose;
    this->enableConvTransposeDecomposeToPhasedConv =
        enableConvTransposeDecomposeToPhasedConv;
    this->enableConvTranspose1dDecomposeToPhasedConv =
        enableConvTranspose1dDecomposeToPhasedConv;
    this->enableInstanceNormDecompose = enableInstanceNormDecompose;
    this->enableGroupNormDecompose = enableGroupNormDecompose;
    this->enableMatmulNBitsDecompose = enableMatmulNBitsDecompose;
    this->enableGroupQueryAttentionDecompose =
        enableGroupQueryAttentionDecompose;
    this->enableSplitToSliceDecompose = enableSplitToSliceDecompose;
    this->enableConcatFuse = enableConcatFuse;
    this->enableGAPToReduceMean = enableGAPToReduceMean;
    this->enableLstmSeqDecompose = enableLstmSeqDecompose;
    this->enableReduceL2Decompose = enableReduceL2Decompose;
    this->enableGatherToSlice = enableGatherToSlice;
    this->enableRotaryEmbeddingRecompose = enableRotaryEmbeddingRecompose;
    this->qdqConstProp = enableQDQConstProp;
    this->enableHardSwishDecompose = enableHardSwishDecompose;
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
          enableHardSwishDecompose);
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

    inferFunctionReturnShapes(f);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createONNXHybridTransformPass(
    bool enableRecomposition, bool enableQuarkQuantizedOpsLegalization,
    bool enableConvTransposeDecompose,
    bool enableConvTransposeDecomposeToPhasedConv,
    bool enableConvTranspose1dDecomposeToPhasedConv,
    bool enableInstanceNormDecompose, bool enableGroupNormDecompose,
    bool enableMatmulNBitsDecompose, bool enableGroupQueryAttentionDecompose,
    bool enableSplitToSliceDecompose, bool enableConcatFuse,
    bool enableGAPToReduceMean, bool enableLstmSeqDecompose,
    bool enableGatherToSlice, bool enableReduceL2Decompose,
    bool enableRotaryEmbeddingRecompose, bool enableQDQConstProp,
    bool enableHardSwishDecompose) {
  return std::make_unique<ONNXHybridTransformPass>(enableRecomposition,
      enableQuarkQuantizedOpsLegalization, enableConvTransposeDecompose,
      enableConvTransposeDecomposeToPhasedConv,
      enableConvTranspose1dDecomposeToPhasedConv, enableInstanceNormDecompose,
      enableGroupNormDecompose, enableMatmulNBitsDecompose,
      enableGroupQueryAttentionDecompose, enableSplitToSliceDecompose,
      enableConcatFuse, enableGAPToReduceMean, enableLstmSeqDecompose,
      enableGatherToSlice, enableReduceL2Decompose,
      enableRotaryEmbeddingRecompose, enableQDQConstProp,
      enableHardSwishDecompose);
}
