// RUN: onnx-mlir-opt --linalg-ops=MatMul --convert-onnx-to-linalg %s -split-input-file | FileCheck %s

// -----

// Test MatMul 2D x 2D
func.func @test_matmul_2d(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x4xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
  
  // CHECK-LABEL: test_matmul_2d
  // CHECK-DAG: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[EMPTY:%.+]] = tensor.empty() : tensor<2x4xf32>
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPTY]] : tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: [[RESULT:%.+]] = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs([[FILLED]] : tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: return [[RESULT]] : tensor<2x4xf32>
}

// -----

// Test MatMul 2D x 2D with different matrix sizes
func.func @test_matmul_different_sizes(%arg0 : tensor<5x10xf32>, %arg1 : tensor<10x3xf32>) -> tensor<5x3xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<5x10xf32>, tensor<10x3xf32>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
  
  // CHECK-LABEL: test_matmul_different_sizes
  // CHECK-DAG: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[EMPTY:%.+]] = tensor.empty() : tensor<5x3xf32>
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPTY]] : tensor<5x3xf32>) -> tensor<5x3xf32>
  // CHECK: [[RESULT:%.+]] = linalg.matmul ins(%arg0, %arg1 : tensor<5x10xf32>, tensor<10x3xf32>) outs([[FILLED]] : tensor<5x3xf32>) -> tensor<5x3xf32>
  // CHECK: return [[RESULT]] : tensor<5x3xf32>
}

// -----

// Test MatMul with square matrices
func.func @test_matmul_square(%arg0 : tensor<4x4xf32>, %arg1 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  
  // CHECK-LABEL: test_matmul_square
  // CHECK-DAG: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[EMPTY:%.+]] = tensor.empty() : tensor<4x4xf32>
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPTY]] : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: [[RESULT:%.+]] = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs([[FILLED]] : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: return [[RESULT]] : tensor<4x4xf32>
}

// -----

// Test that 3D batch matmul is NOT lowered (should remain as onnx.MatMul)
func.func @test_matmul_3d_batch_not_lowered(%arg0 : tensor<2x3x4xf32>, %arg1 : tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
  
  // CHECK-LABEL: test_matmul_3d_batch_not_lowered
  // CHECK-NOT: linalg.matmul
  // CHECK: "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
}

// -----

// Test that 1D x 2D is NOT lowered (should remain as onnx.MatMul)
func.func @test_matmul_1d_2d_not_lowered(%arg0 : tensor<3xf32>, %arg1 : tensor<3x4xf32>) -> tensor<4xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<3xf32>, tensor<3x4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
  
  // CHECK-LABEL: test_matmul_1d_2d_not_lowered
  // CHECK-NOT: linalg.matmul
  // CHECK: "onnx.MatMul"(%arg0, %arg1) : (tensor<3xf32>, tensor<3x4xf32>) -> tensor<4xf32>
}

// -----

// Test that 2D x 1D is NOT lowered (should remain as onnx.MatMul)
func.func @test_matmul_2d_1d_not_lowered(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3xf32>) -> tensor<2xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
  
  // CHECK-LABEL: test_matmul_2d_1d_not_lowered
  // CHECK-NOT: linalg.matmul
  // CHECK: "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
}

