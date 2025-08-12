module attributes {} {
  func.func @main_graph(%a0: tensor<?x?x?xf32>, %a1: tensor<?x?xf32>) -> tensor<?x?x?xf32> attributes {input_names = ["a", "b"], output_names = ["output"]} {
    // bast23: 3d x 2d + 1D
    %0 =  "onnx.MatMul"(%a0, %a1) : (tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

