// RUN: onnx-mlir-opt --split-input-file --selective-shape-inference %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// Test 1: Relu with unranked output gets shape inferred
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_relu_unranked
func.func @test_relu_unranked(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<1x3x8x8xf32>) -> tensor<*xf32>
  %1 = "onnx.Relu"(%0) : (tensor<*xf32>) -> tensor<1x3x8x8xf32>
  return %1 : tensor<1x3x8x8xf32>
}
// First Relu output should now be ranked
// CHECK: "onnx.Relu"(%arg0) : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
// CHECK: "onnx.Relu"({{.*}}) : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>

// -----

//===----------------------------------------------------------------------===//
// Test 2: Already ranked op is unchanged
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_relu_already_ranked
func.func @test_relu_already_ranked(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
  return %0 : tensor<1x3x8x8xf32>
}
// CHECK: "onnx.Relu"(%arg0) : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>

// -----

//===----------------------------------------------------------------------===//
// Test 3: Cascading shape inference through multiple unranked ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_cascading_inference
func.func @test_cascading_inference(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<1x3x8x8xf32>) -> tensor<*xf32>
  %1 = "onnx.Sigmoid"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  %2 = "onnx.Relu"(%1) : (tensor<*xf32>) -> tensor<1x3x8x8xf32>
  return %2 : tensor<1x3x8x8xf32>
}
// All unranked intermediates should become ranked
// CHECK: "onnx.Relu"(%arg0) : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
// CHECK: "onnx.Sigmoid"({{.*}}) : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
// CHECK: "onnx.Relu"({{.*}}) : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>

// -----

//===----------------------------------------------------------------------===//
// Test 4: XFEConv (ranked) is not touched by the pass
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_xfeconv_skipped
func.func @test_xfeconv_skipped(
    %arg0: tensor<1x8x8x16xf32>,
    %weight: tensor<32x3x3x16xf32>,
    %bias: tensor<32xf32>) -> tensor<1x6x6x32xf32> {
  %0 = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0],
       strides = [1, 1]}
      : (tensor<1x8x8x16xf32>, tensor<32x3x3x16xf32>, tensor<32xf32>)
      -> tensor<1x6x6x32xf32>
  return %0 : tensor<1x6x6x32xf32>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: -> tensor<1x6x6x32xf32>

// -----

//===----------------------------------------------------------------------===//
// Test 5: Mixed - standard ONNX unranked op alongside XFEConv
// Only the standard op should be shape-inferred.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_mixed_standard_and_xfe
func.func @test_mixed_standard_and_xfe(
    %arg0: tensor<1x8x8x16xf32>,
    %weight: tensor<32x3x3x16xf32>,
    %bias: tensor<32xf32>) -> tensor<1x6x6x32xf32> {
  %0 = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0],
       strides = [1, 1]}
      : (tensor<1x8x8x16xf32>, tensor<32x3x3x16xf32>, tensor<32xf32>)
      -> tensor<1x6x6x32xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x6x6x32xf32>) -> tensor<*xf32>
  %2 = "onnx.Relu"(%1) : (tensor<*xf32>) -> tensor<1x6x6x32xf32>
  return %2 : tensor<1x6x6x32xf32>
}
// CHECK: "onnx.XFEConv"
// Relu should be shape-inferred
// CHECK: "onnx.Relu"({{.*}}) : (tensor<1x6x6x32xf32>) -> tensor<1x6x6x32xf32>
// CHECK: "onnx.Relu"({{.*}}) : (tensor<1x6x6x32xf32>) -> tensor<1x6x6x32xf32>

// -----

//===----------------------------------------------------------------------===//
// Test 6: Quantized unranked output gets shape inferred
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_relu_quantized_unranked
func.func @test_relu_quantized_unranked(
    %arg0: tensor<1x3x8x8x!quant.uniform<u8:f32, 0.05:128>>)
    -> tensor<1x3x8x8x!quant.uniform<u8:f32, 0.05:128>> {
  %0 = "onnx.Relu"(%arg0) : (tensor<1x3x8x8x!quant.uniform<u8:f32, 0.05:128>>)
      -> tensor<*x!quant.uniform<u8:f32, 0.05:128>>
  %1 = "onnx.Relu"(%0) : (tensor<*x!quant.uniform<u8:f32, 0.05:128>>)
      -> tensor<1x3x8x8x!quant.uniform<u8:f32, 0.05:128>>
  return %1 : tensor<1x3x8x8x!quant.uniform<u8:f32, 0.05:128>>
}
// CHECK: "onnx.Relu"(%arg0)
// CHECK-SAME: -> tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-02:128>>
// CHECK: "onnx.Relu"({{.*}})
// CHECK-SAME: -> tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-02:128>>

// -----

//===----------------------------------------------------------------------===//
// Test 7: Add with unranked output and broadcast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_add_broadcast_unranked
func.func @test_add_broadcast_unranked(
    %arg0: tensor<1x3x8x8xf32>,
    %arg1: tensor<1x3x1x1xf32>) -> tensor<1x3x8x8xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x3x8x8xf32>, tensor<1x3x1x1xf32>) -> tensor<*xf32>
  %1 = "onnx.Relu"(%0) : (tensor<*xf32>) -> tensor<1x3x8x8xf32>
  return %1 : tensor<1x3x8x8xf32>
}
// CHECK: "onnx.Add"(%arg0, %arg1) : (tensor<1x3x8x8xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x8x8xf32>
// CHECK: "onnx.Relu"({{.*}}) : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
