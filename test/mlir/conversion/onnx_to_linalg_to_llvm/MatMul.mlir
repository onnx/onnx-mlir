// RUN: onnx-mlir-opt --convert-onnx-to-linalg --linalg-to-llvm-pipeline %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: @test_matmul_2d_to_llvm
// CHECK: func.func @test_matmul_2d_to_llvm
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: !llvm.ptr)
// CHECK: llvm.func
func.func @test_matmul_2d_to_llvm(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x4xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

