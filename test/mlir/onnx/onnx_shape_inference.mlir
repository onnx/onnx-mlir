// RUN: onnf-opt --shape-inference %s -split-input-file | FileCheck %s

/// Test the default behavior of transpose when no information for the
/// permutation of the axes is provided.
func @test_default_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_transpose
  // CHECK: [[RES:%.+]] = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
  // CHECK: return [[RES]] : tensor<32x1x5x5xf32>
}

/// Test the shape inferencing scheme for the matmul operation.
/// MatMul: 1-D x 1-D
func @test_matmul_1(%arg0 : tensor<32xf32>, %arg1 : tensor<32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<32xf32>, tensor<32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_1
  // CHECK: [[RES1:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<32xf32>, tensor<32xf32>) -> tensor<1xf32>
  // CHECK: return [[RES1]] : tensor<1xf32>
}

/// MatMul: K-D x 2-D (K > 2)
func @test_matmul_2(%arg0 : tensor<16x?x64x42xf32>, %arg1 : tensor<42x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?x64x42xf32>, tensor<42x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_2
  // CHECK: [[RES2:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?x64x42xf32>, tensor<42x32xf32>) -> tensor<16x?x64x32xf32>
  // CHECK: return [[RES2]] : tensor<16x?x64x32xf32>
}

/// MatMul: 2-D x K-D (K > 2)
func @test_matmul_3(%arg0 : tensor<64x42xf32>, %arg1 : tensor<16x?x42x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<16x?x42x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_3
  // CHECK: [[RES3:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<16x?x42x32xf32>) -> tensor<16x?x64x32xf32>
  // CHECK: return [[RES3]] : tensor<16x?x64x32xf32>
}

/// MatMul: 2-D x K-D (K > 2)
func @test_matmul_4(%arg0 : tensor<64x42xf32>, %arg1 : tensor<?x?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<?x?x?x?xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_4
  // CHECK: [[RES4:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x64x?xf32>
  // CHECK: return [[RES4]] : tensor<?x?x64x?xf32>
}