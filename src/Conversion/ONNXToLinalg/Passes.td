#ifndef ONNX_MLIR_CONVERSION_ONNX_TO_LINALG_PASSES
#define ONNX_MLIR_CONVERSION_ONNX_TO_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def ConvertONNXToLinalg : Pass<"convert-onnx-to-linalg", "func::FuncOp"> {
  let summary = "Lower ONNX operations to Linalg dialect";
  let description = [{
    This pass converts ONNX operations to Linalg dialect operations.
    It applies lowering patterns to transform ONNX operations into their
    equivalent Linalg operations.
  }];
  let options = [
    Option<"linalgOps", "linalg-ops", "std::string", /*default=*/"",
           "Specify which ONNX operations should be lowered to Linalg dialect. "
           "Operations are specified as a comma-separated list or regex patterns. "
           "Example: --convert-onnx-to-linalg='linalg-ops=MatMul,Conv' or "
           "--convert-onnx-to-linalg='linalg-ops=\"MatMul.*\"'. "
           "Special values: ALL (all operations), NONE (no operations). "
           "If not specified, defaults to converting all operations.">,
    Option<"useLinalgPath", "use-linalg-path", "bool", /*default=*/"false",
           "Enable Linalg path for all operations (equivalent to --use-linalg-path "
           "in onnx-mlir).">
  ];
}

#endif
