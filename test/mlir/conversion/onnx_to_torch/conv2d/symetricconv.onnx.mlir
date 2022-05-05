//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x4x4x4xf32> attributes {input_names = ["input.1"], output_names = ["9"]} {
    %0 = "onnx.Constant"() {value = dense<[[[[-0.397202909]], [[-0.36456573]], [[0.554807603]]], [[[0.128473148]], [[-0.0792445242]], [[0.0802431181]]], [[[0.43963322]], [[-0.535440087]], [[-0.211081728]]], [[[0.205662608]], [[-0.0686611608]], [[0.00833704136]]], [[[0.115851991]], [[0.450539321]], [[0.0498450212]]], [[[0.0374790728]], [[0.0478444695]], [[0.0221564472]]], [[[0.135107309]], [[0.0581301674]], [[0.0429676324]]], [[[-0.0938993394]], [[0.55964911]], [[-0.168949708]]]]> : tensor<8x3x1x1xf32>} : () -> tensor<8x3x1x1xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.212501466, 0.139968663, -0.53322506, -0.286446363, 0.449779421, -0.505617738, 0.116871089, -0.292490959]> : tensor<8xf32>} : () -> tensor<8xf32>
//CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
//CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int1, %int1{{_*[0-9]*}} :
//CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int0, %int0{{_*[0-9]*}} :
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], onnx_node_name = "Conv_0", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<8x3x1x1xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
//CHECK: torch.aten.conv2d %arg0, %2, %3, [[STRIDE]], [[PAD]], [[DILATION]], %int1{{_*[0-9]*}} : 
    %3 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<8x8x3x3xf32>} : () -> tensor<8x8x3x3xf32>
    %4 = "onnx.Constant"() {value = dense<[0.112499714, 0.0667099878, -0.0583707429, -0.0186368581, 0.0810219943, -0.0844935328, 0.0696898848, -0.0807781293]> : tensor<8xf32>} : () -> tensor<8xf32>
    %5 = "onnx.Conv"(%2, %3, %4) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_1", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x8x8x8xf32>, tensor<8x8x3x3xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
    %6 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<4x8x7x7xf32>} : () -> tensor<4x8x7x7xf32>
    %7 = "onnx.Constant"() {value = dense<[0.0238167867, -0.00796119682, 0.0322854742, -0.0237512067]> : tensor<4xf32>} : () -> tensor<4xf32>
    %8 = "onnx.Conv"(%5, %6, %7) {dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], onnx_node_name = "Conv_2", pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x8x8x8xf32>, tensor<4x8x7x7xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
    return %8 : tensor<1x4x4x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 8 , 8] , \22name\22 : \22input.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 4 , 4 , 4] , \22name\22 : \229\22 }\0A\0A]\00"} : () -> ()
}
