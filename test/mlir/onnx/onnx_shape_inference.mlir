// RUN: onnf-opt --shape-inference %s -split-input-file | FileCheck %s

/// Test the default behavior of transpose when no information for the
/// permutation of the axes is provided.
func @test_default_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_default_transpose
// CHECK: [[RES:%.+]] = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
// CHECK: return [[RES]] : tensor<32x1x5x5xf32>