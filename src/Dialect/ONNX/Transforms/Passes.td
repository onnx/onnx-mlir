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

#endif
