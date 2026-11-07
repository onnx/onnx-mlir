module {
  func.func @main_graph(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x4x2x3xi64> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} :
        (tensor<1x2x3x4xf32>) -> tensor<1x3x4x2xf32>
    %1 = "onnx.Cast"(%0) {to = i64} :
        (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xi64>
    %2 = "onnx.Transpose"(%1) {perm = [0, 2, 3, 1]} :
        (tensor<1x3x4x2xi64>) -> tensor<1x4x2x3xi64>
    onnx.Return %2 : tensor<1x4x2x3xi64>
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
