//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s

module {
  func @main_graph(%arg0: tensor<20x16x48x38xf32>) -> tensor<20x16x48x38xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    //CHECK: [[ALPHA:%[^ ]*]] = torch.constant.float 0.10000000
    //CHECK: torch.aten.leaky_relu %arg0, [[ALPHA]] : !torch.vtensor<[20,16,48,38],f32>, !torch.float -> !torch.vtensor<[20,16,48,38],f32>
    %2 = "onnx.LeakyRelu"(%arg0) {alpha = 1.000000e-01 : f32, max_input0_bitwidth = 1.600000e+01 : f32, max_input0_narrow = 0 : si64, max_input0_rounding_mode = "ROUND", max_input0_scale_factor = 9.765625E-4 : f32, max_input0_signed = 1 : si64, max_input1_bitwidth = 1.600000e+01 : f32, max_input1_narrow = 0 : si64, max_input1_rounding_mode = "ROUND", max_input1_scale_factor = 9.765625E-4 : f32, max_input1_signed = 1 : si64, max_output_bitwidth = 8.000000e+00 : f32, max_output_narrow = 0 : si64, max_output_rounding_mode = "ROUND", max_output_scale_factor = 2.500000e-01 : f32, max_output_signed = 1 : si64, mul_input0_bitwidth = 1.600000e+01 : f32, mul_input0_narrow = 0 : si64, mul_input0_rounding_mode = "ROUND", mul_input0_scale_factor = 9.765625E-4 : f32, mul_input0_signed = 1 : si64, mul_input1_bitwidth = 1.600000e+01 : f32, mul_input1_narrow = 0 : si64, mul_input1_rounding_mode = "ROUND", mul_input1_scale_factor = 1.90734863E-6 : f32, mul_input1_signed = 0 : si64, mul_output_bitwidth = 1.600000e+01 : f32, mul_output_narrow = 0 : si64, mul_output_rounding_mode = "ROUND", mul_output_scale_factor = 9.765625E-4 : f32, mul_output_signed = 1 : si64, onnx_node_name = "LeakyRelu_0"} : (tensor<20x16x48x38xf32>) -> tensor<20x16x48x38xf32>
    return %2 : tensor<20x16x48x38xf32>
  }
}
