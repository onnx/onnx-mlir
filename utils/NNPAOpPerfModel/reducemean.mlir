// need to define e1=1
module attributes {} {
  func.func @main_graph(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> attributes {input_names = ["x"], output_names = ["output"]} {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %0 : tensor<?x?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

