#ifndef ONNX_MLIR_ONNX_TRANSFORMS_PASSES
#define ONNX_MLIR_ONNX_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def AppendDecodingStrategyPass : Pass<"append-decoding-strategy", "ModuleOp"> {
  let summary = "Append a decoding strategy such as greedy algorithm, top_k, top_p, to the end of the main graph.";
  let description = [{
    This pass is to append a decoding strategy such as greedy algorithm, top_k, top_p, to the end of the main graph.
    This is mainly used for decoder models to embed the decoding strategy into the compiled model.
    The decoding strategy consumes the output of the original model (logits) and produces the next token.
  }];
}

def FusionOpTransformPass : Pass<"fusion-op-transform", "ModuleOp"> {
  let summary = "Fuse chains of ONNX operations into a single ONNXFusedOp (non-accelerator-specific kinds)";
  let description = [{
    Detects and replaces chains of ONNX operations with a single ONNXFusedOp,
    for fusion kinds that are not accelerator-specific -- see
    src/Dialect/ONNX/Transforms/ONNXFusionOpHelper.hpp for the
    FusionOpKindHelper subclasses this pass registers. A no-op whenever this
    pass isn't scheduled at all, which is the case under any accelerator
    (e.g. NNPA) that instead merges these same patterns into its own, later
    fusion pass -- see src/Compiler/CompilerPasses.cpp's `targetCPU` gate.
  }];
}

def AppendInstrumentedOutputsPass : Pass<"append-instrumented-outputs", "ModuleOp"> {
  let summary = "append tensors matched by onnx_node_name as extra outputs of the entry function(s)";
  let description = [{
    This pass appends selected intermediate tensors, matched by onnx_node_name
    using the same "REGEX[:inN+outN]" syntax as --instrument-onnx-node, as extra
    outputs of each entry function. Unlike --instrument-onnx-node (which prints
    matched tensors as "==SIG-REPORT==" text at runtime), this lets tooling built
    around comparing model outputs (e.g. utils/RunONNXModel.py's
    --save-ref/--verify=ref) diff them directly, with no log parsing required.
  }];
}

def ClampSoftmaxMaskPass : Pass<"clamp-softmax-mask", "func::FuncOp"> {
  let summary = "Clamp large negative mask constants that only feed a Softmax.";
  let description = [{
    Models often mask attention by adding a large negative value to the
    Softmax input, built from a constant such as -FLT_MAX (e.g.
    torch.finfo(float32).min) or -inf. In reduced precision (f16, or
    dlfloat16 on NNPA, max ~8.6e9) such a value leaves no room for the scores
    and masks added to it: the sum overflows, and a row where every entry is
    masked (e.g. a padded query in a sliding-window mask) turns into NaN,
    which then spreads to every position of that batch row.

    Softmax gives the same result when every input below a large negative
    value L is raised to L, as exp(L - rowMax) is exactly 0; only rows that
    are fully masked change, from NaN/overflow to uniform weights (what fp32
    gives for -FLT_MAX). L is getAttentionMaskFillValue() (-1e9 for f32, -1e4
    for f16), the value onnx.Attention's decomposition uses for its masks.

    The pass starts from each sink: the input of onnx.Softmax, and the
    attn_mask input of onnx.Attention. It walks back through ops that cannot
    make a value more negative than their inputs (Add, Sub's first operand,
    the two value inputs of Where, Min, Max, Concat, Expand, Reshape,
    Unsqueeze, Squeeze, Transpose, Slice, Gather's data, Identity, and Cast
    between identical types). Every float onnx.Constant reached this way that
    holds a value below L gets those values set to L, provided all its
    transitive uses stay within these walks (so a -FLT_MAX used as a sentinel
    elsewhere is left unchanged). It never adds operations.
  }];

}

def ExpandAttentionMaskPass : Pass<"expand-attention-mask", "func::FuncOp"> {
  let summary = "Expand attention mask and scale tensors in attention patterns for NNPA.";
  let description = [{
    Identifies attention mask and scale tensors in MatMul-[Mul]-Add-Softmax 
    patterns where Add (and optionally Mul) perform broadcasting. The Mul 
    operation is optional and, when present, must have a scalar tensor as its 
    second operand for attention scaling.
    
    Expands mask and/or scale operands once if:
    1. Add is part of MatMul-[Mul]-Add-Softmax pattern
    2. Operand is used in ≥4 such patterns (configurable threshold)
    3. All uses have same target shape (verified using DimAnalysis)
    
    Eliminates broadcasting in Add and Mul operations for NNPA compatibility.
  }];
  
  let constructor = "onnx_mlir::createExpandAttentionMaskPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::ONNXDialect"];
}

#endif
