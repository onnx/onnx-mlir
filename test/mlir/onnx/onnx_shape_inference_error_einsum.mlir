// RUN: onnx-mlir-opt --shape-inference %s -split-input-file -verify-diagnostics

func @test_einsum_invalid_equation_syntax(%arg0: tensor<f32>) -> tensor<*xf32> {
  // expected-error @+1 {{invalid equation syntax}}
  %0 = "onnx.Einsum"(%arg0) {equation = "-"} : (tensor<f32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @test_einsum_output_subscript_doesnt_appear_in_inputs(%arg0: tensor<f32>) -> tensor<*xf32> {
  // expected-error @+1 {{output subscript i doesn't appear in inputs}}
  %0 = "onnx.Einsum"(%arg0) {equation = "->i"} : (tensor<f32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @test_einsum_repeated_subscript_in_output(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{subscript i appears multiple times in the output}}
  %0 = "onnx.Einsum"(%arg0) {equation = "ij->ii"} : (tensor<2x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
