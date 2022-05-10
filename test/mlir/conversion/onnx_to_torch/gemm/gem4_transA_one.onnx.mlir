//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x3x50x50xf32>) -> tensor<f32> attributes {input_names = ["input.1"], output_names = ["9"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<25x3x5x5xf32>} : () -> tensor<25x3x5x5xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.0579525419, -0.0102146557, -0.0799950659, 0.0750127658, -0.00767843611, -0.0696612075, -0.0935630947, 0.0982276946, 0.07648395, -0.0547179356, 0.0544863939, -0.0709260106, 0.045096375, -0.0340108834, 0.00953918881, -0.0945071503, -0.00908934511, -0.0524844788, -0.00781042967, 0.0216633677, 0.0876768603, 0.0462757275, -0.0837277248, 0.0623709745, 0.0593513921]> : tensor<25xf32>} : () -> tensor<25xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Conv_0", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x3x50x50xf32>, tensor<25x3x5x5xf32>, tensor<25xf32>) -> tensor<1x25x25x25xf32>
    %3 = "onnx.MaxPoolSingleOut"(%2) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_1", pads = [0, 0, 0, 0], strides = [4, 4]} : (tensor<1x25x25x25xf32>) -> tensor<1x25x6x6xf32>
    %4 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<1x900xf32>} : () -> tensor<1x900xf32>
    %5 = "onnx.Constant"() {value = dense<0.00692393398> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK: %int[[AVAL:[^ ]*]] = torch.constant.int 0
//CHECK: %int[[BVAL:[^ ]*]] = torch.constant.int 1
//CHECK: torch.aten.transpose.int %13, %int[[AVAL:[^ ]*]], %int[[BVAL:[^ ]*]] :
    %6 = "onnx.Gemm"(%3, %4, %5) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_3", transA = 1 : si64} : (tensor<1x25x6x6xf32>, tensor<1x900xf32>, tensor<1xf32>) -> tensor<1x1xf32>
    %7 = "onnx.ReduceMean"(%6) {keepdims = 0 : si64, onnx_node_name = "ReduceMean_4"} : (tensor<1x1xf32>) -> tensor<f32>
    return %7 : tensor<f32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 50 , 50] , \22name\22 : \22input.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [] , \22name\22 : \229\22 }\0A\0A]\00"} : () -> ()
}
