//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<2x7xf32>, %arg1: tensor<7x4xf32>, %arg2: tensor<1x4xf32>) -> tensor<2x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK: %[[TRANSB:.*]] = torch.constant.int 1
//CHECK: [[BETA:%[^ ]*]] = torch.constant.float 5.000000e-01
//CHECK: torch.aten.mul.Scalar %arg2, [[BETA]] : !torch.vtensor<[1,4],f32>, !torch.float -> !torch.vtensor<[2,4],f32>
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {beta = 5.000000e-01 : f32} : (tensor<2x7xf32>, tensor<7x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
//CHECK: torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[2,7],f32>, !torch.vtensor<[7,4],f32> -> !torch.vtensor<[2,4],f32>
//CHECK: torch.aten.add.Tensor %7, %6, %[[TRANSB]] : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
    return %0 : tensor<2x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [2 , 7] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [7 , 4] , \22name\22 : \22b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 4] , \22name\22 : \22c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [2 , 4] , \22name\22 : \22y\22 }\0A\0A]\00"} : () -> ()
}
