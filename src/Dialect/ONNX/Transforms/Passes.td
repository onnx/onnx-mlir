/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===- Passes.td - ONNX Transform Passes -------------------*- tablegen -*-===//
//
// Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ONNXDecomposePatternOptions {
  list<Option> options = [
    Option<"target", "target", "std::string", /*default=*/"\"\"",
           "Target dialect to decompose into (e.g. stablehlo); "
           "empty string means the default ONNX-to-ONNX decomposition.">,

    Option<"enableConvTransposeDecompose", "enable-convtranspose",
           "bool", /*default=*/"false",
           "Enable decomposition of ConvTranspose.">,

    Option<"enableConvTransposeDecomposeToPhasedConv",
           "enable-convtranspose-phased",
           "bool", /*default=*/"false",
           "Enable decomposition of ONNX ConvTranspose operator to "
           "4 phased Conv.">,

    Option<"enableConvTranspose1dDecomposeToPhasedConv",
           "enable-convtranspose-1d-phased",
           "bool", /*default=*/"false",
           "Enable decomposition of ONNX ConvTranspose 1D operator to "
           "phased Conv.">,

    Option<"enableInstanceNormDecompose", "enable-instancenorm-decompose",
           "bool", /*default=*/"true",
           "Enable decomposition of InstanceNormalization to "
           "LayerNormalization.">,

    Option<"enableGroupNormDecompose", "enable-groupnorm-decompose",
           "bool", /*default=*/"true",
           "Enable decomposition of GroupNormalization to "
           "LayerNormalization.">,

    Option<"enableMatmulNBitsDecompose", "enable-matmulnbits-decompose",
           "bool", /*default=*/"false",
           "Enable decomposition of Microsoft MatmulNBits to "
           "DequantizeLinear and Matmul ops.">,

    Option<"enableGroupQueryAttentionDecompose",
           "enable-groupqueryattention-decompose",
           "bool", /*default=*/"true",
           "Enable decomposition of Microsoft GroupQueryAttention to "
           "onnx.Attention and onnx.RotaryEmbedding ops.">,

    Option<"enableSplitToSliceDecompose", "enable-split-to-slice",
           "bool", /*default=*/"false",
           "Enable decomposition of Split to Slice operations.">,

    Option<"enableConcatFuse", "enable-concat-fuse",
           "bool", /*default=*/"true",
           "Enable ConcatFusePattern rewriter.">,

    Option<"enableLstmSeqDecompose", "enable-lstm-seq-decomposition",
           "bool", /*default=*/"false",
           "Enable sequence-length decomposition of LSTM (unroll a "
           "seq_len>1 LSTM into a chain of seq_len=1 LSTMs).">,

    Option<"enableReduceL2Decompose", "enable-reducel2-decompose",
           "bool", /*default=*/"true",
           "Enable decomposition of ReduceL2 to Sqrt(ReduceSumSquare(x)).">,

    Option<"enableGatherToSlice", "enable-gather-to-slice",
           "bool", /*default=*/"true",
           "Enable decomposition of Gather with scalar index to "
           "Slice+Reshape.">,

    Option<"enableHardSwishDecompose", "enable-hardswish-decompose",
           "bool", /*default=*/"true",
           "Enable decomposition of HardSwish into "
           "x * HardSigmoid(x) (alpha=1/6, beta=0.5).">,

    Option<"enableGroupQueryAttentionCacheSlicing",
           "enable-groupqueryattention-cache-slicing",
           "bool", /*default=*/"true",
           "Enable slicing of cos/sin caches during GroupQueryAttention "
           "decomposition. Set to false to keep the cache and synthesize "
           "position_ids instead.">,

    Option<"enableSeparatePhasedConvsForConvTranspose",
           "enable-separate-phased-convs-for-convtranspose",
           "bool", /*default=*/"false",
           "In 4-phase ConvTranspose decomposition, force 4 separate Conv ops "
           "(concatenated) instead of combining the phased weights into a "
           "single Conv, when the conv output channels are DMA-aligned.">
  ];
}

def ONNXHybridTransformOptions {
  list<Option> options = [
    Option<"shapeInference", "shape-inference",
           "bool", /*default=*/"true",
           "Enable shape inference in hybrid transform.">,
    Option<"canonicalization", "canonicalization",
           "bool", /*default=*/"true",
           "Enable canonicalization in hybrid transform.">,
    Option<"constantPropagation", "constant-propagation",
           "bool", /*default=*/"true",
           "Enable constant propagation in hybrid transform.">,
    Option<"qdqConstProp", "qdq-const-prop",
           "bool", /*default=*/"false",
           "Enable constant propagation for QDQ.">,
    Option<"decomposition", "decomposition",
           "bool", /*default=*/"true",
           "Enable decomposition in hybrid transform.">,
    Option<"recomposition", "recomposition",
           "bool", /*default=*/"true",
           "Enable recomposition in hybrid transform.">,
    Option<"quarkQuantizedOpsLegalization",
           "quark-quantized-ops-legalization",
           "bool", /*default=*/"false",
           "Enable legalization of quark-quantized operations from F32 to BF16.">,
    Option<"maxNumRewritesOffset", "max-num-rewrites-offset",
           "int", /*default=*/"20",
           "Rewrites limit: -1 means no limit, otherwise added to "
           "func #ops * max-num-rewrites-multiplier.">,
    Option<"maxNumRewritesMultiplier", "max-num-rewrites-multiplier",
           "float", /*default=*/"0.2",
           "Rewrites limit factor.">,
    Option<"enableGAPToReduceMean", "enable-globalaveragepool-to-reducemean",
           "bool", /*default=*/"true",
           "Enable canonicalize from GlobalAveragePool to ReduceMean.">,
    Option<"enableRotaryEmbeddingRecompose",
           "enable-rotary-embedding-recompose",
           "bool", /*default=*/"false",
           "Recompose LlamaRotaryEmbedding style RoPE into "
           "onnx.RotaryEmbedding.">
  ];
}

def DecomposeONNXToONNXPass : Pass<"decompose-onnx", "::mlir::func::FuncOp"> {
  let summary = "Decompose ONNX operations into composition of other ONNX "
                "operations.";
  let description = [{
    This pass rewrites ONNX operations into equivalent compositions of simpler
    ONNX operations.  It is applied before shape inference so there is no
    expectation of ranked tensor types at this point.

    Individual decompositions can be enabled or disabled via the options below.
    The pass-level defaults match the programmatic defaults used by
    createDecomposeONNXToONNXPass().
  }];

  let options = ONNXDecomposePatternOptions.options;
}

def ONNXHybridTransformPass
    : Pass<"onnx-hybrid-transform", "::mlir::func::FuncOp"> {
  let summary = "Run shape inference, canonicalization, constant propagation, "
                "decomposition, and recomposition together.";
  let description = [{
    This pass applies ONNX-to-ONNX transforms with one greedy rewrite driver.
    Decomposition-related options are shared with the decompose-onnx pass.
  }];

  let options =
      ONNXHybridTransformOptions.options # ONNXDecomposePatternOptions.options;
}
