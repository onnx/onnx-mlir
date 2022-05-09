//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x4xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<"0xDEADBEEF"> : tensor<2x3x5x4xf32>} : () -> tensor<2x3x5x4xf32>
//CHECK: torch.aten.matmul %arg0, %2 : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[2,3,5,4],f32> -> !torch.vtensor<[2,3,4,4],f32> 
 %1 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "MatMul_1"} : (tensor<2x3x4x5xf32>, tensor<2x3x5x4xf32>) -> tensor<2x3x4x4xf32>
    return %1 : tensor<2x3x4x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [2 , 3 , 4 , 5] , \22name\22 : \220\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [2 , 3 , 4 , 4] , \22name\22 : \222\22 }\0A\0A]\00"} : () -> ()
}
