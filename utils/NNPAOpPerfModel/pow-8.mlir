module attributes {} {
  func.func @main_graph(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> attributes {input_names = ["x"], output_names = ["output"]} {
    %cst = onnx.Constant dense<8.0> : tensor<f32>
    %0 = "onnx.Pow"(%arg0, %cst) : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
