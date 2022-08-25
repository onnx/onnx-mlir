//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<2x3xf32>) -> tensor<2x5xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<0.0> : tensor<3x5xf32>} : () -> tensor<3x5xf32>
//CHECK: torch.aten.matmul %arg0, %{{[^,]*}} : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,5],f32> -> !torch.vtensor<[2,5],f32>
    %1 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "MatMul_1"} : (tensor<2x3xf32>, tensor<3x5xf32>) -> tensor<2x5xf32>
    return %1 : tensor<2x5xf32>
  }
}
