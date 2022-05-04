//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<2x3x5x7xf32>) -> tensor<6x35xf32> attributes {input_names = ["0"], output_names = ["1"]} {
    //CHECK: %[[FLAT1:.*]] = torch.aten.flatten.using_ints %arg0, %int0, %int2 : !torch.vtensor<[2,3,5,7],f32>, !torch.int, !torch.int -> !torch.vtensor<[6,5,7],f32>
    //CHECK: %[[FLAT2:.*]] = torch.aten.flatten.using_ints %[[FLAT1]], %int1, %int-1 : !torch.vtensor<[6,5,7],f32>, !torch.int, !torch.int -> !torch.vtensor<[6,35],f32>
    %0 = "onnx.Flatten"(%arg0) {axis = 2 : si64, onnx_node_name = "Flatten_0"} : (tensor<2x3x5x7xf32>) -> tensor<6x35xf32>
    return %0 : tensor<6x35xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 4 , 4] , \22name\22 : \220\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 16] , \22name\22 : \221\22 }\0A\0A]\00"} : () -> ()
}
