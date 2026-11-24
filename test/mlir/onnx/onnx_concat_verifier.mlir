// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

func.func @concat_all_empty(
    %arg0: tensor<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64} :
      (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>
  return %0 : tensor<0xf32>
}

// -----

func.func @concat_all_empty_axis_out_of_range(
    %arg0: tensor<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  // expected-error @+1 {{onnx.Concat: 'axis' value is 1, accepted range is [-1, 0]}}
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} :
      (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>
  return %0 : tensor<0xf32>
}

// -----

func.func @concat_no_inputs() -> tensor<*xf32> {
  // expected-error @+1 {{requires at least one input}}
  %0 = "onnx.Concat"() {axis = 0 : si64} : () -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
