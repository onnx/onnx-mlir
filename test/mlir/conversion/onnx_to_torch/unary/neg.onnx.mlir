//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x3x22x22xf32>) -> tensor<1x4x4x4xf32> attributes {input_names = ["input.1"], output_names = ["7"]} {
    %0 = "onnx.Constant"() {value = dense<1.0> : tensor<10x3x5x5xf32>} : () -> tensor<10x3x5x5xf32>
    %1 = "onnx.Constant"() {value = dense<[0.093399994, 0.062870495, -0.00694241608, -0.10948731, 0.073852092, -0.109910354, -0.0999641194, -6.2124664E-4, -0.0370393321, -0.0176933873]> : tensor<10xf32>} : () -> tensor<10xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x3x22x22xf32>, tensor<10x3x5x5xf32>, tensor<10xf32>) -> tensor<1x10x10x10xf32>
    %3 = "onnx.Constant"() {value = dense<1.0> : tensor<4x10x3x3xf32>} : () -> tensor<4x10x3x3xf32>
    %4 = "onnx.Constant"() {value = dense<[-0.0907176434, 0.0795172527, 0.00843401439, -0.101543337]> : tensor<4xf32>} : () -> tensor<4xf32>
    %5 = "onnx.Conv"(%2, %3, %4) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_1", pads = [2, 2, 2, 2], strides = [3, 3]} : (tensor<1x10x10x10xf32>, tensor<4x10x3x3xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    %6 = "onnx.Neg"(%5) {onnx_node_name = "Neg_2"} : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    return %6 : tensor<1x4x4x4xf32>
//CHECK: torch.aten.neg  %{{[^,]*}}: !torch.vtensor<[1,4,4,4],f32> -> !torch.vtensor<[1,4,4,4],f32>
  }
}
