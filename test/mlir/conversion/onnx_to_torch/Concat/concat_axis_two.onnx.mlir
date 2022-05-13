//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x4x22x22xf32>) -> tensor<1x17x36x18xf32> attributes {input_names = ["input"], output_names = ["4"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<17x4x7x7xf32>} : () -> tensor<17x4x7x7xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.00891791097, 0.0335937142, 0.0285940785, -0.0106998943, 6.851200e-02, 0.0444460176, -0.0343254209, 0.0408403724, -0.00486376463, 0.0107753109, -0.0287600346, -0.0574335977, 0.0697360784, 0.0415690467, 0.00419284636, -0.0293454193, 0.0533983484]> : tensor<17xf32>} : () -> tensor<17xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x4x22x22xf32>, tensor<17x4x7x7xf32>, tensor<17xf32>) -> tensor<1x17x18x18xf32>
//CHECK: %int[[DIM:.]] = torch.constant.int 2
//CHECK: [[INPUT:%.]] = torch.prim.ListConstruct %8, %8 : (!torch.vtensor<[1,17,18,18],f32>, !torch.vtensor<[1,17,18,18],f32>) -> !torch.list<vtensor<[1,17,18,18],f32>>
    %3 = "onnx.Concat"(%2, %2) {axis = 2 : si64, onnx_node_name = "Concat_1"} : (tensor<1x17x18x18xf32>, tensor<1x17x18x18xf32>) -> tensor<1x17x36x18xf32>
//CHECK: torch.aten.cat [[INPUT:%.]], %int[[DIM:.]] : !torch.list<vtensor<[1,17,18,18],f32>>, !torch.int -> !torch.vtensor<[1,17,36,18],f32>
    return %3 : tensor<1x17x36x18xf32>
  }
}
