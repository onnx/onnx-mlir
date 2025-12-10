// Test MatMul conversion from ONNX to Linalg to LLVM IR
// This file tests the full pipeline: ONNX -> Linalg -> Affine -> LLVM IR

module {
  func.func @main_graph(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

