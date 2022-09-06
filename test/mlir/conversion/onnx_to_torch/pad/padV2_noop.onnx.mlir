//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s

module {
  func @main_graph(%arg0: tensor<20x16x44x32xf32>) -> tensor<20x16x48x38xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    // CHECK-NOT: torch.aten.constant_pad_nd %arg0, [*], [*] : !torch.vtensor<[20,16,44,32],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[20,16,48,38],f32>
    // CHECK: return %arg0
    %0 = "onnx.PadV2"(%arg0) {ifm_bitwidth = 8.000000e+00 : f32, ifm_narrow = 0 : si64, ifm_rounding_mode = "ROUND", ifm_scale_factor = 2.500000e-01 : f32, ifm_signed = 1 : si64, mode = "constant", onnx_node_name = "Pad_0", pads = [0, 0, 0, 0, 0, 0, 0, 0], value = 0xFF800000 : f32} : (tensor<20x16x44x32xf32>) -> tensor<20x16x48x38xf32>
    return %0 : tensor<20x16x48x38xf32>
  }
}