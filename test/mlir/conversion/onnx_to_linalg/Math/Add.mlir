// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=onnx.Add' %s -split-input-file | FileCheck %s

// -----
// Same-shape ranked tensors should lower.
func.func @test_add_same_shape(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>)
    -> tensor<2x3xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>

  // CHECK-LABEL: test_add_same_shape
  // CHECK: linalg.generic
}

// -----
// Broadcasting not supported initially: mismatched shapes should stay ONNX.
func.func @test_add_reject_broadcast(%arg0: tensor<2x3xf32>, %arg1: tensor<1x3xf32>)
    -> tensor<2x3xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>

  // CHECK-LABEL: test_add_reject_broadcast
  // CHECK-NOT: linalg.generic
  // CHECK: "onnx.Add"
}

