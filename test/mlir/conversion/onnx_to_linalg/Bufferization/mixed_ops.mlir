// RUN: onnx-mlir-opt --convert-onnx-to-linalg --one-shot-bufferize=allow-unknown-ops %s -split-input-file | FileCheck %s

// Test that One-Shot Bufferization works with mixed Linalg and ONNX operations.
// MatMul is converted to Linalg, while Add remains as ONNX.
// Bufferization should succeed and automatically insert bufferization.to_tensor
// casts to bridge between memref (Linalg) and tensor (ONNX) types.

// -----

// Test: MatMul (converted to Linalg) + Add (remains as ONNX)
func.func @test_matmul_add_mixed(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x4xf32>, %arg2 : tensor<2x4xf32>) -> tensor<2x4xf32> {
  // MatMul will be converted to linalg.matmul (produces tensor)
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  // Add remains as ONNX (expects tensor)
  %1 = "onnx.Add"(%0, %arg2) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
  
  // CHECK-LABEL: func.func @test_matmul_add_mixed
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<2x3xf32>, %[[ARG1:.*]]: tensor<3x4xf32>, %[[ARG2:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32>
  // After bufferization, MatMul is converted to linalg.matmul (uses memref internally)
  // CHECK: bufferization.to_buffer
  // CHECK: linalg.fill
  // CHECK: linalg.matmul
  // CHECK-SAME: ins({{.*}} : memref<2x3xf32{{.*}}>, memref<3x4xf32{{.*}}>) outs({{.*}} : memref<2x4xf32>)
  // CHECK: bufferization.to_tensor
  // CHECK: "onnx.Add"
  // CHECK: return
}

// -----

// Test: Multiple ONNX operations chained (all remain as ONNX)
func.func @test_onnx_chain(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>, %arg2 : tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
  
  // CHECK-LABEL: func.func @test_onnx_chain
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<2x4xf32>, %[[ARG1:.*]]: tensor<2x4xf32>, %[[ARG2:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32>
  // All operations should remain as ONNX (not converted to Linalg)
  // CHECK: "onnx.Add"
  // CHECK: "onnx.Add"
  // CHECK: return
}

