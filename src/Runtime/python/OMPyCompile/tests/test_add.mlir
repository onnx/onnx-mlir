module {
  func.func @main_graph(%arg0: tensor<3x4x5xf32> {onnx.name = "x"}, %arg1: tensor<3x4x5xf32> {onnx.name = "y"}) -> (tensor<3x4x5xf32> {onnx.name = "sum"}) {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
    onnx.Return %0 : tensor<3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
