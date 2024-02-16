// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_unsqueeze(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[0, 3]> : tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueeze
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 10, 10, 1], strides: [100, 10, 1, 1] : memref<10x10xf32> to memref<1x10x10x1xf32>
  // CHECK: return [[RES]] : memref<1x10x10x1xf32>
}

// -----

func.func private @test_unsqueezev11(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) {axes=[0,3]} : (tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueezev11
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 10, 10, 1], strides: [100, 10, 1, 1] : memref<10x10xf32> to memref<1x10x10x1xf32>
  // CHECK: return [[RES]] : memref<1x10x10x1xf32>
}

