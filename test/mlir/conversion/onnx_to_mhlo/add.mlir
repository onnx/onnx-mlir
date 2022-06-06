// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s -split-input-file | FileCheck %s

// CHECK-LABEL:  func @test_add
func.func @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK: mhlo.add
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}