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

// Options common to both decompose-onnx and recompose-onnx: same C++ member,
// same meaning. Declared once and pulled into both passes (and the hybrid pass)
// so the generated options struct never gets a duplicate member.
def SharedDecomposeRecomposePatternOptions {
  list<Option> options = [
    Option<"target", "target", "std::string", /*default=*/"\"\"",
           "Target dialect to decompose/recompose into (e.g. stablehlo); "
           "empty string means the default ONNX-to-ONNX behavior.">,
    Option<"enableDepthToSpaceDecompose",
           "enable-depthtospace-decompose",
           "bool", /*default=*/"false",
           "Decompose onnx.DepthToSpace (DCR and CRD) into "
           "Reshape/Transpose/Reshape so backends without a DepthToSpace "
           "kernel can run it.">
  ];
}

def ONNXDecomposePatternOptions {
  list<Option> options = [
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

    Option<"holdBackPreallocatedGQADecompose",
           "hold-back-preallocated-gqa-decompose",
           "bool", /*default=*/"false",
           "Skip GroupQueryAttention decomposition on preallocated full-RoPE "
           "nodes, leaving them intact for a whole-node match. "
           "Empty-cache prefill and partial-RoPE nodes are still decomposed "
           "here, where the resulting ONNX ops can still be lowered.">,

    Option<"enableGQAUint16CacheSlotRewrite",
           "enable-gqa-uint16-cache-slot-rewrite",
           "bool", /*default=*/"false",
           "Rewrite preallocated GroupQueryAttention cache slot updates using "
           "a uint16 selector and arithmetic instead of Expand/ScatterElements. Warning: this is not safe if the value cannot fit into uint16 range.">,

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

    Option<"enableLstmDecompose", "enable-lstm-decomposition",
           "bool", /*default=*/"false",
           "Enable decomposition of static LSTM operations into basic ONNX "
           "operations and bounded onnx.Loop operations.">,

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

    Option<"enableSeparatePhasedConvsForConvTranspose",
           "enable-separate-phased-convs-for-convtranspose",
           "bool", /*default=*/"false",
           "In 4-phase ConvTranspose decomposition, force 4 separate Conv ops "
           "(concatenated) instead of combining the phased weights into a "
           "single Conv, when the conv output channels are DMA-aligned.">,

    Option<"enableConvTransposeDecomposeToDepthToSpace",
           "enable-convtranspose-depthtospace",
           "bool", /*default=*/"false",
           "In the phased ConvTranspose decomposition, emit a DepthToSpace "
           "(DCR) as the final interleave instead of "
           "Reshape/Transpose/Reshape.">,

    Option<"enableConvTransposeToResize",
           "convert-convtranspose-to-resize",
           "bool", /*default=*/"false",
           "Rewrite a nearest-neighbor upsampling ConvTranspose (kernel==stride, "
           "pads=0, per-channel all-ones weights, group=1 block-diagonal or "
           "depthwise group=C) into onnx.Resize (nearest mode) and keep it out "
           "of the phased-Conv ConvTranspose decomposition.">,

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

  let options = SharedDecomposeRecomposePatternOptions.options
              # ONNXDecomposePatternOptions.options;
}

// Recompose-specific options. The shared `target` option lives in
// SharedDecomposeRecomposePatternOptions.
def ONNXRecomposePatternOptions {
  list<Option> options = [
    Option<"enableRotaryEmbeddingRecompose", "enable-rotary-embedding-recompose",
           "bool", /*default=*/"false",
           "Recompose HuggingFace-style RoPE elementwise subgraphs "
           "(rotate_half + 2 muls + 1 add) into onnx.RotaryEmbedding">,

    Option<"enableReduceL2Recompositions",
           "enable-reducel2-recompositions",
           "bool", /*default=*/"false",
           "Recompose ReduceL2 from Sqrt/Pow(0.5) square-reduction "
           "chains into ReduceL2. Off by default.">
  ];
}

def RecomposeONNXToONNXPass : Pass<"recompose-onnx", "::mlir::func::FuncOp"> {
  let summary = "Recompose ONNX operations into composition of other ONNX "
                "operations.";
  let description = [{
    This pass rewrites ONNX operations into equivalent compositions of higher level
    ONNX operations.  It is applied before shape inference so there is no
    expectation of ranked tensor types at this point.

    Individual decompositions can be enabled or disabled via the options above.
    The pass-level defaults match the programmatic defaults used by
    createRecomposeONNXToONNXPass().
  }];

  let options = SharedDecomposeRecomposePatternOptions.options
              # ONNXRecomposePatternOptions.options;
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
    Option<"quantConstFold", "quant-const-fold",
           "bool", /*default=*/"false",
           "Enable folding of a constant QuantizeLinear into an integer "
           "constant.">,
    Option<"dequantConstFold", "dequant-const-fold",
           "bool", /*default=*/"false",
           "Enable folding of a constant DequantizeLinear into a float "
           "constant.">,
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
  ];
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
      ONNXHybridTransformOptions.options # SharedDecomposeRecomposePatternOptions.options # ONNXDecomposePatternOptions.options # ONNXRecomposePatternOptions.options;

  let genOptionsParser = 1;
}

def XFEONNXOpsetVerifierPass
    : Pass<"xfe-onnx-opset-verifier", "::mlir::func::FuncOp"> {
  let summary = "Verify XFE ONNX opset constraints.";
  let description = [{
    Verifies that the ops inside a function match the XFE ONNX opset
    constraints. Currently implements a check for disallowed ops.
  }];
  let options = [
    Option<"disallowedOps", "disallowed-ops", "std::string",
           /*default=*/"\"Flatten,Squeeze,Unsqueeze\"",
           "Comma-separated bare ONNX op names that must not appear. Skips ops with non-static shapes.">,
    Option<"verifyNonNegativeAxis", "verify-non-negative-axis", "bool",
           /*default=*/"true",
           "Verify ranked ONNX ops do not carry negative constant axis/axes values.">
  ];
}

def DedupDQsPass : Pass<"dedup-dqs", "::mlir::func::FuncOp"> {
  let summary = "Alternative to CSE, targets only DQ ops";
}

def ConstPropONNXToONNXPass : Pass<"constprop-onnx", "::mlir::func::FuncOp"> {
  let summary = "ConstProp ONNX operations into composition of other ONNX operations.";
  let options = [
    Option<"enableQDQ", "enable-qdq", "bool", "false",
           "Enable QDQ constant propagation.">,
    Option<"enableQuantConstFold", "enable-quant-const-fold", "bool", "false",
           "Enable constant folding of quantized operations.">,
    Option<"enableDequantConstFold", "enable-dequant-const-fold", "bool", "false",
           "Enable constant folding of dequantized operations.">,
    Option<"maxLoopUnrollCount", "max-loop-unroll-count", "int64_t", "64",
           "Maximum constant onnx.Loop trip count to unroll.">
  ];
}

def ConvertToChannelLastPass
    : Pass<"convert-to-channel-last", "::mlir::func::FuncOp"> {
  let summary = "Convert ONNX operations to ChannelLast variants with transposes";
  let options = [
    ListOption<"whitelist", "whitelist", "std::string",
               "ONNX operation names to convert, e.g. onnx.Conv. Converts every supported op when empty.">
  ];
}

def QDQCanonicalizePass : Pass<"qdq-canonicalize", "::mlir::func::FuncOp"> {
  let summary = "Canonicalize ONNX quantize and dequantize operations.";
  let options = [
    Option<"removeBinary", "remove-binary", "bool", "false",
           "Remove redundant QDQ around binary operations.">,
    Option<"removeQDQAroundOps", "remove-qdq-around-ops", "bool", "false",
           "Remove redundant QDQ around operations.">,
    Option<"maxRoundTripDiff", "max-round-trip-diff", "int64_t", "0",
           "Maximum absolute difference allowed between an input integer and its DQ->Q output, checked over the full storage range. 0 requires bit-exact scale and zero-point; >0 tolerates near-equal parameters.">
  ];
}
