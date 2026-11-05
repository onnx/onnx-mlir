// RUN: onnx-mlir-opt --shape-inference %s -split-input-file -verify-diagnostics | FileCheck %s

func.func @split_opset18_uneven_num_outputs(
    %arg0: tensor<2x10xf32>) ->
    (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "onnx.Split"(%arg0, %none) {
    axis = 1 : si64, num_outputs = 3 : si64
  } : (tensor<2x10xf32>, none) ->
      (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1, %0#2 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: func.func @split_opset18_uneven_num_outputs(
// CHECK: %{{.*}}:3 = "onnx.Split"{{.*}} -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x2xf32>)

// -----

func.func @split_opset18_rejects_negative_last_output(
    %arg0: tensor<2x5xf32>) ->
    (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+3 {{'num_outputs' produces a negative size for the last output}}
  // expected-error @+2 {{Failed to scan parameters successfully}}
  // expected-error @+1 {{shape inference failed}}
  %0:4 = "onnx.Split"(%arg0, %none) {
    axis = 1 : si64, num_outputs = 4 : si64
  } : (tensor<2x5xf32>, none) ->
      (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1, %0#2, %0#3 :
      tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
}

// -----

func.func @split_opset18_even_num_outputs(
    %arg0: tensor<2x12xf32>) ->
    (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "onnx.Split"(%arg0, %none) {
    axis = 1 : si64, num_outputs = 3 : si64
  } : (tensor<2x12xf32>, none) ->
      (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1, %0#2 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: func.func @split_opset18_even_num_outputs(
// CHECK: %{{.*}}:3 = "onnx.Split"{{.*}} -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>)

// -----

func.func @split_opset18_one_output(
    %arg0: tensor<2x10xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Split"(%arg0, %none) {
    axis = 1 : si64, num_outputs = 1 : si64
  } : (tensor<2x10xf32>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func.func @split_opset18_one_output(
// CHECK: %{{.*}} = "onnx.Split"{{.*}} -> tensor<2x10xf32>

// -----

func.func @split_opset18_dynamic_dimension(
    %arg0: tensor<2x?xf32>) ->
    (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "onnx.Split"(%arg0, %none) {
    axis = 1 : si64, num_outputs = 3 : si64
  } : (tensor<2x?xf32>, none) ->
      (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1, %0#2 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: func.func @split_opset18_dynamic_dimension(
// CHECK: %{{.*}}:3 = "onnx.Split"{{.*}} -> (tensor<2x?xf32>, tensor<2x?xf32>, tensor<2x?xf32>)

// -----

func.func @split_opset18_zero_dimension(
    %arg0: tensor<2x0xf32>) ->
    (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "onnx.Split"(%arg0, %none) {
    axis = 1 : si64, num_outputs = 3 : si64
  } : (tensor<2x0xf32>, none) ->
      (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1, %0#2 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: func.func @split_opset18_zero_dimension(
// CHECK: %{{.*}}:3 = "onnx.Split"{{.*}} -> (tensor<2x0xf32>, tensor<2x0xf32>, tensor<2x0xf32>)

// -----

func.func @split_opset13_equal(
    %arg0: tensor<2x12xf32>) ->
    (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "onnx.SplitV13"(%arg0, %none) {axis = 1 : si64} :
      (tensor<2x12xf32>, none) ->
      (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1, %0#2 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: func.func @split_opset13_equal(
// CHECK: %{{.*}}:3 = "onnx.SplitV13"{{.*}} -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>)

// -----

func.func @split_opset13_explicit_sizes(
    %arg0: tensor<2x12xf32>) ->
    (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %split = "onnx.Constant"() {
    value = dense<[2, 3, 7]> : tensor<3xi64>
  } : () -> tensor<3xi64>
  %0:3 = "onnx.SplitV13"(%arg0, %split) {axis = 1 : si64} :
      (tensor<2x12xf32>, tensor<3xi64>) ->
      (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1, %0#2 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: func.func @split_opset13_explicit_sizes(
// CHECK: %{{.*}}:3 = "onnx.SplitV13"{{.*}} -> (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x7xf32>)

// -----

func.func @split_opset13_rejects_uneven_equal_split(
    %arg0: tensor<2x10xf32>) ->
    (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+3 {{The dimension at the split axis is expected to be divisible by the number of results}}
  // expected-error @+2 {{Failed to scan parameters successfully}}
  // expected-error @+1 {{shape inference failed}}
  %0:3 = "onnx.SplitV13"(%arg0, %none) {axis = 1 : si64} :
      (tensor<2x10xf32>, none) ->
      (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1, %0#2 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
}
