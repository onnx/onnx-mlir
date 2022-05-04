//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x38x19x19xf32>) -> tensor<1x38x19x19xf32> attributes {input_names = ["input.1"], output_names = ["8"]} {
    //CHECK: %[[DIM:.*]] = torch.constant.int 3
    //CHECK: %[[TRUE:.*]] = torch.constant.bool true
    %7 = "onnx.Softmax"(%arg0) {axis = 3 : si64, onnx_node_name = "Softmax_3", onnx_opset = 9 : si64} : (tensor<1x38x19x19xf32>) -> tensor<1x38x19x19xf32>
    //CHECK: torch.aten._softmax %arg0, %[[DIM]], %[[TRUE]] : !torch.vtensor<[1,38,19,19],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,38,19,19],f32>
    return %7 : tensor<1x38x19x19xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 75 , 75] , \22name\22 : \22input.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 38 , 19 , 19] , \22name\22 : \228\22 }\0A\0A]\00"} : () -> ()
}
