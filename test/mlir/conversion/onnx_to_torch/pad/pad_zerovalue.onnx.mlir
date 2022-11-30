//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @main_graph(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x16x12xf32> attributes {input_names = ["input.1"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 2, 2, 0, 0, 2, 2]> : tensor<8xi64>} : () -> tensor<8xi64>
    %1 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK: [[ALPHA:%[^ ]*]] = torch.constant.float 0.000000e+00
//CHECK: torch.aten.constant_pad_nd %arg0, %{{[^,]*}}, [[ALPHA]] {layer_name = "Pad_0"} : !torch.vtensor<[1,1,5,5],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,1,9,9],f32>
    %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant", onnx_node_name = "Pad_0"} : (tensor<1x1x5x5xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x1x9x9xf32>
    %3 = "onnx.Constant"() {value = dense<[0, 0, 3, 1, 0, 0, 4, 2]> : tensor<8xi64>} : () -> tensor<8xi64>
    %4 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK: torch.aten.constant_pad_nd  %{{[^,]*}},  %{{[^,]*}}, [[ALPHA]] {layer_name = "Pad_0"} : !torch.vtensor<[1,1,9,9],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,1,16,12],f32>
    %5 = "onnx.Pad"(%2, %3, %4) {mode = "constant", onnx_node_name = "Pad_0"} : (tensor<1x1x9x9xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x1x16x12xf32>
    return %5 : tensor<1x1x16x12xf32>
  }
}
