// RUN: onnx-mlir-opt --onnx-cse-with-node-name %s -split-input-file | FileCheck %s

// COM: Test CSE with multiple identical operations - all should be merged to first.
func.func @test_multiple_cse(%arg0: tensor<5x5xf32>) -> (tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>) {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "relu1"} : (tensor<5x5xf32>) -> tensor<5x5xf32>
  %1 = "onnx.Relu"(%arg0) {onnx_node_name = "relu2"} : (tensor<5x5xf32>) -> tensor<5x5xf32>
  %2 = "onnx.Relu"(%arg0) {onnx_node_name = "relu3"} : (tensor<5x5xf32>) -> tensor<5x5xf32>
  return %0, %1, %2 : tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>
}

// CHECK-LABEL: func.func @test_multiple_cse
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<5x5xf32>)
// CHECK-NEXT:    %[[RELU:.*]] = "onnx.Relu"(%[[ARG0]]) {onnx_node_name = "relu1"} : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK-NEXT:    return %[[RELU]], %[[RELU]], %[[RELU]] : tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>

// -----

// COM: Test CSE with operations that have no node names initially.
func.func @test_cse_no_node_names(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %0 = "onnx.Add"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %1 = "onnx.Add"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0, %1 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: func.func @test_cse_no_node_names
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<10xf32>)
// CHECK-NEXT:    %[[ADD:.*]] = "onnx.Add"(%[[ARG0]], %[[ARG0]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    return %[[ADD]], %[[ADD]] : tensor<10xf32>, tensor<10xf32>

// -----

// COM: Test CSE with mixed operations - some with node names, some without.
func.func @test_cse_mixed_node_names(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) {
  %0 = "onnx.Add"(%arg0, %arg0) {onnx_node_name = "add1"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %1 = "onnx.Add"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %2 = "onnx.Add"(%arg0, %arg0) {onnx_node_name = "add3"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0, %1, %2 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: func.func @test_cse_mixed_node_names
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<10xf32>)
// CHECK-NEXT:    %[[ADD:.*]] = "onnx.Add"(%[[ARG0]], %[[ARG0]]) {onnx_node_name = "add1"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    return %[[ADD]], %[[ADD]], %[[ADD]] : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>

// -----

// COM: Test that operations with different inputs are not merged.
func.func @test_no_cse_different_inputs(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "relu1"} : (tensor<10xf32>) -> tensor<10xf32>
  %1 = "onnx.Relu"(%arg1) {onnx_node_name = "relu2"} : (tensor<10xf32>) -> tensor<10xf32>
  return %0, %1 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: func.func @test_no_cse_different_inputs
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<10xf32>, %[[ARG1:.*]]: tensor<10xf32>)
// CHECK-NEXT:    %[[RELU1:.*]] = "onnx.Relu"(%[[ARG0]]) {onnx_node_name = "relu1"} : (tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %[[RELU2:.*]] = "onnx.Relu"(%[[ARG1]]) {onnx_node_name = "relu2"} : (tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    return %[[RELU1]], %[[RELU2]] : tensor<10xf32>, tensor<10xf32>

// -----

// COM: Test CSE with a chain of operations.
func.func @test_cse_chain(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %0 = "onnx.Add"(%arg0, %arg0) {onnx_node_name = "add1"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "relu1"} : (tensor<10xf32>) -> tensor<10xf32>
  %2 = "onnx.Add"(%arg0, %arg0) {onnx_node_name = "add2"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %3 = "onnx.Relu"(%2) {onnx_node_name = "relu2"} : (tensor<10xf32>) -> tensor<10xf32>
  return %1, %3 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: func.func @test_cse_chain
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<10xf32>)
// CHECK-NEXT:    %[[ADD:.*]] = "onnx.Add"(%[[ARG0]], %[[ARG0]]) {onnx_node_name = "add1"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %[[RELU:.*]] = "onnx.Relu"(%[[ADD]]) {onnx_node_name = "relu1"} : (tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    return %[[RELU]], %[[RELU]] : tensor<10xf32>, tensor<10xf32>
