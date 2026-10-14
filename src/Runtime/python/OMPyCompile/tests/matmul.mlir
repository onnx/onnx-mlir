module {
  func.func @main_graph(%arg0 : tensor<?x?x768xf32>, %arg1 : tensor<768x768xf32>) -> tensor<?x?x768xf32> {
    %r = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
    onnx.Return %r : tensor<?x?x768xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
