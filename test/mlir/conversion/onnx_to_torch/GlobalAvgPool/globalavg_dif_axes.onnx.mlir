//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x5x1x1xf32> attributes {input_names = ["input.1"], output_names = ["8"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<10x3x3x3xf32>} : () -> tensor<10x3x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<[0.0234757643, 0.155270293, 0.0288646873, 0.128271028, 0.0515015386, 0.132100895, -0.00874967314, -0.138507202, -0.0193597451, -0.113981798]> : tensor<10xf32>} : () -> tensor<10xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x3x8x8xf32>, tensor<10x3x3x3xf32>, tensor<10xf32>) -> tensor<1x10x4x4xf32>
    %3 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<5x10x3x3xf32>} : () -> tensor<5x10x3x3xf32>
    %4 = "onnx.Constant"() {value = dense<[-0.0370794684, -3.994020e-02, -3.261460e-02, -0.103588313, -0.0369978286]> : tensor<5xf32>} : () -> tensor<5xf32>
    %5 = "onnx.Conv"(%2, %3, %4) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_1", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x10x4x4xf32>, tensor<5x10x3x3xf32>, tensor<5xf32>) -> tensor<1x5x4x4xf32>
//CHECK: %[[DIM:int0_20]] = torch.constant.int 0
    %6 = "onnx.ReduceMean"(%5) {axes = [2, 1]} : (tensor<1x5x4x4xf32>) -> tensor<1x5x1x1xf32>
//CHECK: torch.aten.mean %15, %[[DIM]] : !torch.vtensor<[1,5,4,4],f32>, !torch.int -> !torch.vtensor<[1,5,1,1],f32>
//CHECK: %[[DIM1:int0_23]] = torch.constant.int 0
//CHECK: torch.aten.mean %17, %[[DIM1]] : !torch.vtensor<[1,5,1,1],f32>, !torch.int -> !torch.vtensor<[1,5,1,1],f32>
    %7 = "onnx.ReduceMean"(%6) {axes = [2, 3]} : (tensor<1x5x1x1xf32>) -> tensor<1x5x1x1xf32>

    return %7 : tensor<1x5x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 8 , 8] , \22name\22 : \22input.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 5 , 1 , 1] , \22name\22 : \228\22 }\0A\0A]\00"} : () -> ()
}
