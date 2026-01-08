// RUN: onnx-mlir-opt --convert-onnx-to-linalg
// --one-shot-bufferize=allow-unknown-ops --convert-onnx-to-krnl %s
// -split-input-file | FileCheck %s

// Test the complete pipeline: MatMul(Linalg) + Add(Krnl) with bufferization
// This tests: convert-onnx-to-linalg → one-shot-bufferize →
// convert-onnx-to-krnl

// -----

// Test: Full pipeline - MatMul(Linalg) + Add(Krnl)
func.func @test_full_pipeline(% arg0
                              : tensor<2x3xf32>, % arg1
                              : tensor<3x4xf32>, % arg2
                              : tensor<2x4xf32>)
    ->tensor<2x4xf32> {
  // MatMul will be converted to linalg.matmul
  % 0 = "onnx.MatMul"(% arg0, % arg1)
      : (tensor<2x3xf32>, tensor<3x4xf32>)->tensor<2x4xf32>
        // Add will be converted to Krnl (after bufferization)
        % 1 = "onnx.Add"(% 0, % arg2)
      : (tensor<2x4xf32>, tensor<2x4xf32>)->tensor<2x4xf32> return %
              1 : tensor<2x4xf32>

  // CHECK-LABEL: func.func @test_full_pipeline
  // CHECK-SAME: (%[[ARG0:.*]]: memref<2x3xf32>, %[[ARG1:.*]]: memref<3x4xf32>,
  // %[[ARG2:.*]]: memref<2x4xf32>) -> memref<2x4xf32> After full pipeline:
  // MatMul is linalg.matmul (memref), Add is krnl.iterate (memref) CHECK:
  // linalg.fill CHECK: linalg.matmul CHECK-SAME: ins({{.*}} :
  // memref<2x3xf32{{.*}}>, memref<3x4xf32{{.*}}>) outs({{.*}} :
  // memref<2x4xf32>) CHECK: krnl.define_loops CHECK: krnl.iterate CHECK:
  // krnl.load CHECK: krnl.store CHECK: return
}
