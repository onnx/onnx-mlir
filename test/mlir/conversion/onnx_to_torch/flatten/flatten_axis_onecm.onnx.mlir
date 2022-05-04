//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x4x15x15xf32>) -> tensor<1x5xf32> attributes {input_names = ["input.1"], output_names = ["10"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<6x4x7x7xf32>} : () -> tensor<6x4x7x7xf32>
    %1 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<6xf32>} : () -> tensor<6xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], onnx_node_name = "Conv_0", pads = [2, 2, 2, 2], strides = [3, 3]} : (tensor<1x4x15x15xf32>, tensor<6x4x7x7xf32>, tensor<6xf32>) -> tensor<1x6x5x5xf32>
    %3 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<5x6x5x5xf32>} : () -> tensor<5x6x5x5xf32>
    %4 = "onnx.Constant"() {value = dense<[0.011732677, -0.0649859458, -0.0472838655, -0.0655153841, -0.0196100231]> : tensor<5xf32>} : () -> tensor<5xf32>
    %5 = "onnx.Conv"(%2, %3, %4) {dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Conv_1", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x6x5x5xf32>, tensor<5x6x5x5xf32>, tensor<5xf32>) -> tensor<1x5x2x2xf32>
    %6 = "onnx.MaxPoolSingleOut"(%5) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_2", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x5x2x2xf32>) -> tensor<1x5x1x1xf32>
//CHECK: %int[[START:[^ ]*]] = torch.constant.int 1
//CHECK: %int[[END:.]] = torch.constant.int 4
//CHECK: torch.aten.flatten.using_ints %20, %int[[START:[^ ]*]], %int[[END:.]] : 
    %7 = "onnx.Flatten"(%6) {axis = 1 : si64, onnx_node_name = "Flatten_3"} : (tensor<1x5x1x1xf32>) -> tensor<1x5xf32>
    return %7 : tensor<1x5xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 4 , 15 , 15] , \22name\22 : \22input.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 5] , \22name\22 : \2210\22 }\0A\0A]\00"} : () -> ()
}
