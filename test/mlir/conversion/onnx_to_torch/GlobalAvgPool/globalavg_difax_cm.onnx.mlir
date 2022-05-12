//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x2x15x15xf32>) -> tensor<1x7x1x1xf32> attributes {input_names = ["input.1"], output_names = ["9"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<8x2x5x5xf32>} : () -> tensor<8x2x5x5xf32>
    %1 = "onnx.Constant"() {value = dense<[0.0488194712, 0.0165383574, -0.140125141, 0.0660743564, 0.00143751106, -0.034113694, 0.0405743867, 0.136481673]> : tensor<8xf32>} : () -> tensor<8xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x2x15x15xf32>, tensor<8x2x5x5xf32>, tensor<8xf32>) -> tensor<1x8x7x7xf32>
    %3 = "onnx.MaxPoolSingleOut"(%2) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_1", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x8x7x7xf32>) -> tensor<1x8x3x3xf32>
//CHECK: %[[DIM:int0_15]] = torch.constant.int 0
    %4 = "onnx.ReduceMean"(%3) {axes = [3, 1]} : (tensor<1x8x3x3xf32>) -> tensor<1x8x1x1xf32>
//CHECK: torch.aten.mean %13, %[[DIM]] : !torch.vtensor<[1,8,3,3],f32>, !torch.int -> !torch.vtensor<[1,8,1,1],f32>
    %5 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<7x8x3x3xf32>} : () -> tensor<7x8x3x3xf32>
    %6 = "onnx.Constant"() {value = dense<[0.0882570445, -0.0657474696, -0.0795851946, 0.086938031, -0.0164839271, 0.0901811495, -0.0728597641]> : tensor<7xf32>} : () -> tensor<7xf32>
    %7 = "onnx.Conv"(%4, %5, %6) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_3", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x8x1x1xf32>, tensor<7x8x3x3xf32>, tensor<7xf32>) -> tensor<1x7x1x1xf32>
//CHECK: %[[DIM1:int0_29]] = torch.constant.int 0
    %8 = "onnx.ReduceMean"(%7) {axes = [3, 2]} : (tensor<1x7x1x1xf32>) -> tensor<1x7x1x1xf32>
//CHECK: torch.aten.mean %22, %[[DIM1]] : !torch.vtensor<[1,7,1,1],f32>, !torch.int -> !torch.vtensor<[1,7,1,1],f32>
    return %8 : tensor<1x7x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 2 , 15 , 15] , \22name\22 : \22input.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 7 , 1 , 1] , \22name\22 : \229\22 }\0A\0A]\00"} : () -> ()
}
