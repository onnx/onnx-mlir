#ifndef ONNX_MLIR_CONVERSION_ONNX_TO_LINALG_PASSES
#define ONNX_MLIR_CONVERSION_ONNX_TO_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def ConvertONNXToLinalg : Pass<"convert-onnx-to-linalg", "func::FuncOp"> {
  let summary = "Lower ONNX operations to Linalg dialect";
  let description =
      [{This pass converts ONNX operations to Linalg dialect operations
              .It applies lowering patterns to transform ONNX operations into
                  their equivalent Linalg operations.}];
  let options =
      [Option<"linalgOps", "linalg-ops", "std::string", /*default=*/"",
          "Comma-separated list of ONNX operation names to lower to Linalg. "
          "If empty, all supported operations are lowered. "
          "Example: --linalg-ops=onnx.MatMul,onnx.Add">];
}

#endif
