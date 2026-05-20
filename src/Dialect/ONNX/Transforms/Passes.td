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

#endif
