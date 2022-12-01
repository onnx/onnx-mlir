//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @main_graph(%arg0: tensor<1x2x2x1x2xf32>) -> tensor<2x2x1x2xf32> attributes {input_names = ["input"], output_names = ["output"]} {
    %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
//CHECK: %[[DIM:.*]] = torch.constant.int 0
//CHECK: torch.aten.squeeze.dim %arg0, %[[DIM]] {layer_name = "Squeeze_1"} : !torch.vtensor<[1,2,2,1,2],f32>, !torch.int -> !torch.vtensor<[2,2,1,2],f32>
    %1 = "onnx.Squeeze"(%arg0, %0) {onnx_node_name = "Squeeze_1"} : (tensor<1x2x2x1x2xf32>, tensor<1xi64>) -> tensor<2x2x1x2xf32>
    return %1 : tensor<2x2x1x2xf32>
  }
}
