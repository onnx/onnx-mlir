//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x11x11xf32> attributes {input_names = ["input.1"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>} : () -> tensor<8xi64>
    %1 = "onnx.Constant"() {value = dense<3.500000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
 %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<1x3x8x8xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x3x10x10xf32>
    %3 = "onnx.Constant"() {value = dense<[0, 0, 0, 0, 0, 0, 1, 1]> : tensor<8xi64>} : () -> tensor<8xi64>
    %4 = "onnx.Constant"() {value = dense<5.000000e-01> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK: [[VAL:%[^ ]*]] = torch.constant.float 5.000000e-01
//CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int0, %int1{{_*[0-9]*}}, %int0{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
//CHECK: torch.aten.constant_pad_nd %5, [[PAD]], [[VAL]] : 
   %5 = "onnx.Pad"(%2, %3, %4) {mode = "constant"} : (tensor<1x3x10x10xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x3x11x11xf32>
    return %5 : tensor<1x3x11x11xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 8 , 8] , \22name\22 : \22input.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 11 , 11] , \22name\22 : \222\22 }\0A\0A]\00"} : () -> ()
}
