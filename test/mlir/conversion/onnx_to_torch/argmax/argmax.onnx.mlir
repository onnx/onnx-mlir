module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<4x4xf32>) -> tensor<4xi64> attributes {input_names = ["0"], output_names = ["1"]} {
    %0 = "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 0 : si64, onnx_node_name = "ArgMax_0"} : (tensor<4x4xf32>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [4 , 4] , \22name\22 : \220\22 }\0A\0A]\00@[   { \22type\22 : \22i64\22 , \22dims\22 : [4] , \22name\22 : \221\22 }\0A\0A]\00"} : () -> ()
}
