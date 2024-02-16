// RUN: onnx-mlir-opt --cse %s -split-input-file | FileCheck %s

// COM: Test MLIR CSE pass with ONNX operations to remove common sub-expressions.
func.func @test_cse(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "onnx.Tanh"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  %2 = "onnx.Tanh"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "onnx.Add"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %3 : tensor<*xf32>
  // CHECK-LABEL: test_cse
  // CHECK-NEXT: "onnx.Log"
  // CHECK-COUNT-1: "onnx.Tanh"
  // CHECK-NEXT: "onnx.Add"
}
