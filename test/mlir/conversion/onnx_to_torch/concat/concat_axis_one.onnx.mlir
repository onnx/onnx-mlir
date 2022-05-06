//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<5x5xf32>) -> tensor<5x10xf32> attributes {input_names = ["0"], output_names = ["1"]} {
//CHECK: %int[[DIM:.]] = torch.constant.int 1
//CHECK: [[INPUT:%.]] = torch.prim.ListConstruct %arg0, %arg0 : (!torch.vtensor<[5,5],f32>, !torch.vtensor<[5,5],f32>) -> !torch.list<vtensor<[5,5],f32>>
    %0 = "onnx.Concat"(%arg0, %arg0) {axis = 1 : si64, onnx_node_name = "Concat_0"} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x10xf32>
//CHECK: torch.aten.cat [[INPUT:%.]], %int[[DIM:.]] : !torch.list<vtensor<[5,5],f32>>, !torch.int -> !torch.vtensor<[5,10],f32>
    return %0 : tensor<5x10xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [5 , 5] , \22name\22 : \220\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [5 , 10] , \22name\22 : \221\22 }\0A\0A]\00"} : () -> ()
}
