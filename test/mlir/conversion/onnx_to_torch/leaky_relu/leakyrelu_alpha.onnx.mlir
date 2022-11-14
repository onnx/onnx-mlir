//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @main_graph(%arg0: tensor<5xf32>) -> tensor<5xf32> attributes {input_names = ["input"], output_names = ["1"]} {
//CHECK: [[ALPHA:%[^ ]*]] = torch.constant.float -1.000000e+01
//CHECK: torch.aten.leaky_relu %arg0, [[ALPHA]] {layer_name = "LeakyRelu_0"} : !torch.vtensor<[5],f32>, !torch.float -> !torch.vtensor<[5],f32>
    %0 = "onnx.LeakyRelu"(%arg0) {alpha = -1.000000e+01 : f32, onnx_node_name = "LeakyRelu_0"} : (tensor<5xf32>) -> tensor<5xf32>
    return %0 : tensor<5xf32>
  }
}
