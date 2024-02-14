// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_identity(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_identity
// CHECK: return %arg0 : tensor<10x20x30x40xf32>
}
