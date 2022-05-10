//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module {
  func @main_graph(%arg0: tensor<5xf32>) -> tensor<5xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<1.000000e+01> : tensor<f32>} : () -> tensor<f32>
//CHECK: [[ALPHA:%.]] = torch.vtensor.literal(dense<1.000000e+01> : tensor<f32>) : !torch.vtensor<[],f32>
//CHECK: %int[[OUT:.]] = torch.constant.int 1 
%1 = "onnx.Add"(%arg0, %0) {onnx_node_name = "Add_1"} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
//CHECK: torch.aten.add.Tensor %arg0, [[ALPHA:%.]], %int[[OUT:.]] : !torch.vtensor<[5],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[5],f32>
    return %1 : tensor<5xf32>
  }
}
