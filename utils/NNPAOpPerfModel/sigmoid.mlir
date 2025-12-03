module attributes {} {
  func.func @main_graph(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> attributes {input_names = ["x"], output_names = ["output"]} {
    %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
