// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for UpsampleAndPad operation.
//===----------------------------------------------------------------------===//

// COM: Test with static shapes, strides and pads.
func.func @test_upsample_and_pad_static(%arg0 : tensor<1x3x4x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 2], pads = [1, 1, 1, 1]} : (tensor<1x3x4x4xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_upsample_and_pad_static
  // CHECK: [[RES:%.+]] = "onnx.UpsampleAndPad"(%arg0) <{pads = [1, 1, 1, 1], strides = [2, 2]}> : (tensor<1x3x4x4xf32>) -> tensor<1x3x9x9xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x3x9x9xf32>
}

// -----

// COM: Test with dynamic shapes.
func.func @test_upsample_and_pad_dynamic(%arg0 : tensor<1x3x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 2], pads = [1, 1, 1, 1]} : (tensor<1x3x?x?xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_upsample_and_pad_dynamic
  // CHECK: [[RES:%.+]] = "onnx.UpsampleAndPad"(%arg0) <{pads = [1, 1, 1, 1], strides = [2, 2]}> : (tensor<1x3x?x?xf32>) -> tensor<1x3x?x?xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x3x?x?xf32>
}

// -----

// COM: Test with no padding (pads all zeros).
func.func @test_upsample_no_pad(%arg0 : tensor<2x3x5x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [3, 3], pads = [0, 0, 0, 0]} : (tensor<2x3x5x5xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_upsample_no_pad
  // CHECK: [[RES:%.+]] = "onnx.UpsampleAndPad"(%arg0) <{pads = [0, 0, 0, 0], strides = [3, 3]}> : (tensor<2x3x5x5xf32>) -> tensor<2x3x13x13xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x3x13x13xf32>
}

// -----

// COM: Test with no upsampling (strides all 1).
func.func @test_pad_no_upsample(%arg0 : tensor<1x1x8x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [1, 1], pads = [2, 2, 2, 2]} : (tensor<1x1x8x8xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_pad_no_upsample
  // CHECK: [[RES:%.+]] = "onnx.UpsampleAndPad"(%arg0) <{pads = [2, 2, 2, 2], strides = [1, 1]}> : (tensor<1x1x8x8xf32>) -> tensor<1x1x12x12xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x1x12x12xf32>
}

// -----

// COM: Test with asymmetric padding.
func.func @test_upsample_and_pad_asymmetric(%arg0 : tensor<1x2x3x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 2], pads = [1, 2, 3, 4]} : (tensor<1x2x3x3xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_upsample_and_pad_asymmetric
  // CHECK: [[RES:%.+]] = "onnx.UpsampleAndPad"(%arg0) <{pads = [1, 2, 3, 4], strides = [2, 2]}> : (tensor<1x2x3x3xf32>) -> tensor<1x2x9x11xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x2x9x11xf32>
}

// -----

// COM: Test with 1D (k=1).
func.func @test_upsample_and_pad_1d(%arg0 : tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2], pads = [1, 1]} : (tensor<4x8xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_upsample_and_pad_1d
  // CHECK: [[RES:%.+]] = "onnx.UpsampleAndPad"(%arg0) <{pads = [1, 1], strides = [2]}> : (tensor<4x8xf32>) -> tensor<4x17xf32>
  // CHECK: onnx.Return [[RES]] : tensor<4x17xf32>
}

// -----

// COM: Test with 3D (k=3).
func.func @test_upsample_and_pad_3d(%arg0 : tensor<2x3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 2, 2], pads = [1, 1, 1, 1, 1, 1]} : (tensor<2x3x4x5xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_upsample_and_pad_3d
  // CHECK: [[RES:%.+]] = "onnx.UpsampleAndPad"(%arg0) <{pads = [1, 1, 1, 1, 1, 1], strides = [2, 2, 2]}> : (tensor<2x3x4x5xf32>) -> tensor<2x7x9x11xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x7x9x11xf32>
}

// -----

// COM: Test with 5D input and k=3 (first 2 dims unchanged, last 3 upsampled/padded).
func.func @test_upsample_and_pad_5d_k3(%arg0 : tensor<2x3x4x5x6xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 2, 2], pads = [1, 1, 1, 1, 1, 1]} : (tensor<2x3x4x5x6xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_upsample_and_pad_5d_k3
  // CHECK: [[RES:%.+]] = "onnx.UpsampleAndPad"(%arg0) <{pads = [1, 1, 1, 1, 1, 1], strides = [2, 2, 2]}> : (tensor<2x3x4x5x6xf32>) -> tensor<2x3x9x11x13xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x3x9x11x13xf32>
}

// -----

// COM: Test with optional attributes not specified (defaults).
func.func @test_upsample_and_pad_defaults(%arg0 : tensor<2x3x4x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) : (tensor<2x3x4x4xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_upsample_and_pad_defaults
  // CHECK: [[RES:%.+]] = "onnx.UpsampleAndPad"(%arg0) : (tensor<2x3x4x4xf32>) -> tensor<2x3x4x4xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x3x4x4xf32>
}