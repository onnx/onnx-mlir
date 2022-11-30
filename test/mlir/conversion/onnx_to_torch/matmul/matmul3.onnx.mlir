//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @main_graph(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x4xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<"0xDEADBEEF"> : tensor<2x3x5x4xf32>} : () -> tensor<2x3x5x4xf32>
//CHECK: torch.aten.matmul %arg0, %{{[^,]*}} {layer_name = "MatMul_1"} : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[2,3,5,4],f32> -> !torch.vtensor<[2,3,4,4],f32> 
    %1 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "MatMul_1"} : (tensor<2x3x4x5xf32>, tensor<2x3x5x4xf32>) -> tensor<2x3x4x4xf32>
    return %1 : tensor<2x3x4x4xf32>
  }
}
