//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func.func @main_graph(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> attributes {input_names = ["0"], output_names = ["1"]} {
    %0 = "onnx.MatMul"(%arg0, %arg0) {onnx_node_name = "MatMul_0"} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
//CHECK: torch.aten.matmul %arg0, %arg0 {layer_name = "MatMul_0"} : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],f32> -> !torch.vtensor<[3,3],f32>  
  return %0 : tensor<3x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 3] , \22name\22 : \220\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [3 , 3] , \22name\22 : \221\22 }\0A\0A]\00"} : () -> ()
}
