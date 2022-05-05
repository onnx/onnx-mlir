//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x5x10x10xf32>) -> tensor<1x4x2x3xf32> attributes {input_names = ["input.1"], output_names = ["9"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<9x5x3x4xf32>} : () -> tensor<9x5x3x4xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.0848599597, -0.11561133, 0.0635902881, 0.0752899796, 0.0742146298, -0.0947614163, -4.916150e-03, -0.0397511758, -0.0190015137]> : tensor<9xf32>} : () -> tensor<9xf32>
//CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int3{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
//CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int1, %int1{{_*[0-9]*}} :
//CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int0, %int0{{_*[0-9]*}} :
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 4], onnx_node_name = "Conv_0", pads = [0, 0, 0, 0], strides = [3, 1]} : (tensor<1x5x10x10xf32>, tensor<9x5x3x4xf32>, tensor<9xf32>) -> tensor<1x9x3x7xf32>
//CHECK: torch.aten.conv2d %arg0, %2, %3, [[STRIDE]], [[PAD]], [[DILATION]], %int1{{_*[0-9]*}} :
    %3 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<8x9x4x3xf32>} : () -> tensor<8x9x4x3xf32>
    %4 = "onnx.Constant"() {value = dense<[0.0867970064, -0.00181004219, 0.0419989228, -0.0807644724, -0.0775439292, -0.0465273783, -0.00774087477, 0.0898923427]> : tensor<8xf32>} : () -> tensor<8xf32>
    %5 = "onnx.Conv"(%2, %3, %4) {dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 3], onnx_node_name = "Conv_1", pads = [4, 2, 4, 2], strides = [1, 3]} : (tensor<1x9x3x7xf32>, tensor<8x9x4x3xf32>, tensor<8xf32>) -> tensor<1x8x8x3xf32>
    %6 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<4x8x3x5xf32>} : () -> tensor<4x8x3x5xf32>
    %7 = "onnx.Constant"() {value = dense<[0.0330508351, -0.0333069712, -0.0118212132, 0.0356402248]> : tensor<4xf32>} : () -> tensor<4xf32>
    %8 = "onnx.Conv"(%5, %6, %7) {dilations = [3, 1], group = 1 : si64, kernel_shape = [3, 5], onnx_node_name = "Conv_2", pads = [1, 2, 1, 2], strides = [3, 1]} : (tensor<1x8x8x3xf32>, tensor<4x8x3x5xf32>, tensor<4xf32>) -> tensor<1x4x2x3xf32>
    return %8 : tensor<1x4x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 5 , 10 , 10] , \22name\22 : \22input.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 4 , 2 , 3] , \22name\22 : \229\22 }\0A\0A]\00"} : () -> ()
}
