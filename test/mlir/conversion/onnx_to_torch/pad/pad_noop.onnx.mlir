//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s

module {
  func.func @main_graph(%arg0: tensor<20x16x44x32xf32>) -> tensor<20x16x44x32xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    // CHECK-NOT: torch.aten.constant_pad_nd %arg0, [*], [*] : !torch.vtensor<[20,16,44,32],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[20,16,48,38],f32>
    // CHECK: return %arg0
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<0> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[0.0000]> : tensor<1xf32>} : () -> tensor<1xf32> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {input0_bitwidth = 8.000000e+00 : f32, input0_narrow = 0 : si64, input0_rounding_mode = "ROUND", input0_scale_factor = 1.250000e-01 : f32, input0_signed = 1 : si64, mode = "constant", onnx_node_name = "Pad_0"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<20x16x44x32xf32> 
    return %2 : tensor<20x16x44x32xf32>
  }
}