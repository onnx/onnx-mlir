//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<2x3xf32>) -> tensor<2x5xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {onnx_node_name = "Constant_0", value = dense<[[1.02372718, 0.427999198, 0.286124706, -0.464347422, -0.141982809], [1.41210735, 0.97357428, -0.753101885, 0.476612091, 0.0947996154], [-0.560755193, -8.928040e-01, 1.43628597, 1.49760377, 6.427180e-01]]> : tensor<3x5xf32>} : () -> tensor<3x5xf32>
//CHECK: torch.aten.matmul %arg0, %2 :
    %1 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "MatMul_1"} : (tensor<2x3xf32>, tensor<3x5xf32>) -> tensor<2x5xf32>
    return %1 : tensor<2x5xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [2 , 3] , \22name\22 : \220\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [2 , 5] , \22name\22 : \222\22 }\0A\0A]\00"} : () -> ()
}
