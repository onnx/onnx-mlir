//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<2x1x2x1x2xf32>) -> tensor<2x2x1x2xf32> attributes {input_names = ["input"], output_names = ["output"]} {
    %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
//CHECK: %[[DIM:.*]] = torch.constant.int 1
    %1 = "onnx.Squeeze"(%arg0, %0) {onnx_node_name = "Squeeze_1"} : (tensor<2x1x2x1x2xf32>, tensor<1xi64>) -> tensor<2x2x1x2xf32>
//CHECK: torch.aten.squeeze.dim %arg0, %[[DIM]] : !torch.vtensor<[2,1,2,1,2],f32>, !torch.int -> !torch.vtensor<[2,2,1,2],f32>
    return %1 : tensor<2x2x1x2xf32>
  }
}
