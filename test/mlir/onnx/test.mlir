func @test_elementwise_op_with_scalar_values_1(%arg0 : tensor<1xf32>) -> tensor<*xf32> {
    %0 = "onnx.Cast"(%arg0) {to = 1 : i64} : (tensor<1xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  }

  func @test_elementwise_op_with_scalar_values_2(%arg0 : tensor<1xi32>) -> tensor<*xf32> {
    %0 = "onnx.Cast"(%arg0) {to = 1 : i64} : (tensor<1xi32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  }