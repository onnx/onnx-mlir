// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_reshape_constant(%arg0 : tensor<1x10xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[2, 5]> : tensor<2xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<1x10xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL:     test_reshape_constant
// CHECK: krnl.global
// CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2, 5], strides: [5, 1] : memref<1x10xf32> to memref<2x5xf32>
// CHECK: return [[RES]] : memref<2x5xf32>
}
