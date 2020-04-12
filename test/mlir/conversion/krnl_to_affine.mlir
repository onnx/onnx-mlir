// RUN: onnx-mlir-opt --lower-krnl %s -split-input-file | FileCheck %s

func @test_lower_degenerate_iterate(%arg0: memref<f32>) -> memref<f32> {
  %0 = alloc() : memref<f32>
  krnl.iterate() with () {
    %1 = load %arg0[] : memref<f32>
    store %1, %0[] : memref<f32>
  }
  return %0 : memref<f32>
  // CHECK-LABEL: test_lower_degenerate_iterate
  // CHECK-NEXT: [[ALLOC:%.+]] = alloc() : memref<f32>
  // CHECK-NEXT: [[LOAD:%.+]] = load %{{.*}}[] : memref<f32>
  // CHECK-NEXT: store [[LOAD]], [[ALLOC]][] : memref<f32>
  // CHECK-NEXT: return [[ALLOC]] : memref<f32>
}