//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x11x11xf32> attributes {input_names = ["input.1"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>} : () -> tensor<8xi64>
    %1 = "onnx.Constant"() {value = dense<3.500000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK-DAG: [[ALPHA1:%[^ ]*]] = torch.constant.float 3.500000e+00
//CHECK-DAG: [[ALPHA2:%[^ ]*]] = torch.constant.float 5.000000e-01
//CHECK: torch.aten.constant_pad_nd %arg0, %{{[^,]*}}, [[ALPHA1]] : !torch.vtensor<[1,3,8,8],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,3,10,10],f32>
    %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<1x3x8x8xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x3x10x10xf32>
    %3 = "onnx.Constant"() {value = dense<[0, 0, 0, 0, 0, 0, 1, 1]> : tensor<8xi64>} : () -> tensor<8xi64>
    %4 = "onnx.Constant"() {value = dense<5.000000e-01> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK: torch.aten.constant_pad_nd  %{{[^,]*}},  %{{[^,]*}}, [[ALPHA2]] : !torch.vtensor<[1,3,10,10],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,3,11,11],f32>
    %5 = "onnx.Pad"(%2, %3, %4) {mode = "constant"} : (tensor<1x3x10x10xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x3x11x11xf32>
    return %5 : tensor<1x3x11x11xf32>
  }
}
