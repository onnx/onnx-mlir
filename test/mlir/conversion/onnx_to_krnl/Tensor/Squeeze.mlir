// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[1, -2]> : tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeeze
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 32, 64], strides: [2048, 64, 1] : memref<16x1x32x1x64xf32> to memref<16x32x64xf32>
  // CHECK: return [[RES]] : memref<16x32x64xf32>
}

// -----

func.func private @test_squeezev11(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) { axes = [1, -2]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeezev11
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 32, 64], strides: [2048, 64, 1] : memref<16x1x32x1x64xf32> to memref<16x32x64xf32>
  // CHECK: return [[RES]] : memref<16x32x64xf32>
}

