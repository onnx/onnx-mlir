//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x16x12xf32> attributes {input_names = ["input.1"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 2, 2, 0, 0, 2, 2]> : tensor<8xi64>} : () -> tensor<8xi64>
    %1 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK: [[VAL:%[^ ]*]] = torch.constant.float 0.000000e+00
//CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int2, %int2{{_*[0-9]*}}, %int2{{_*[0-9]*}}, %int2{{_*[0-9]*}} :
//CHECK: torch.aten.constant_pad_nd %arg0, [[PAD]], [[VAL]] :
    %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<1x1x5x5xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x1x9x9xf32>
    %3 = "onnx.Constant"() {value = dense<[0, 0, 3, 1, 0, 0, 4, 2]> : tensor<8xi64>} : () -> tensor<8xi64>
    %4 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
    %5 = "onnx.Pad"(%2, %3, %4) {mode = "constant"} : (tensor<1x1x9x9xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x1x16x12xf32>
    return %5 : tensor<1x1x16x12xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 5 , 5] , \22name\22 : \22input.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 16 , 12] , \22name\22 : \222\22 }\0A\0A]\00"} : () -> ()
}
