// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// RUN: onnx-mlir-opt --conv-with-bias %s | FileCheck %s

// Test 1: Basic Conv + Add -> Conv with bias (quantized)
// CHECK-LABEL: @conv_add_to_conv_bias
func.func @conv_add_to_conv_bias(%arg0: tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>> {
    %weights = onnx.Constant {value = dense<1> : tensor<4x3x3x3xi8>} : tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>
    %none = "onnx.NoValue"() {value} : () -> none
    %conv = "onnx.Conv"(%arg0, %weights, %none) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>, none) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>

    // Add a constant bias (4 channels) - broadcast compatible shape
    %bias = onnx.Constant {value = dense<[[[[2]], [[4]], [[6]], [[8]]]]> : tensor<1x4x1x1xi8>} : tensor<1x4x1x1x!quant.uniform<i8:f32, 5.000000e-01>>
    %result = "onnx.Add"(%conv, %bias) : (tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<1x4x1x1x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>

    return %result : tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>
}
// CHECK-NOT: onnx.Add
// CHECK: onnx.Conv
// CHECK-NOT: none

// -----

// Test 2: Add with constant on left side (quantized)
// CHECK-LABEL: @conv_add_reversed
func.func @conv_add_reversed(%arg0: tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>> {
    %weights = onnx.Constant {value = dense<1> : tensor<4x3x3x3xi8>} : tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>
    %none = "onnx.NoValue"() {value} : () -> none
    %conv = "onnx.Conv"(%arg0, %weights, %none) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>, none) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>

    // Add with constant on the left - broadcast compatible shape
    %bias = onnx.Constant {value = dense<1> : tensor<1x4x1x1xi8>} : tensor<1x4x1x1x!quant.uniform<i8:f32, 5.000000e-01>>
    %result = "onnx.Add"(%bias, %conv) : (tensor<1x4x1x1x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>

    return %result : tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>
}
// CHECK-NOT: onnx.Add
// CHECK: onnx.Conv
// CHECK-NOT: none

// -----

// Test 3: Conv already has bias - should NOT match (quantized)
// CHECK-LABEL: @conv_with_existing_bias
func.func @conv_with_existing_bias(%arg0: tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>> {
    %weights = onnx.Constant {value = dense<1> : tensor<4x3x3x3xi8>} : tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>
    %existing_bias = onnx.Constant {value = dense<1> : tensor<4xi8>} : tensor<4x!quant.uniform<i8:f32, 5.000000e-01>>
    %conv = "onnx.Conv"(%arg0, %weights, %existing_bias) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>

    // This add should NOT be folded since conv already has bias
    %extra_bias = onnx.Constant {value = dense<[[[[2]], [[4]], [[6]], [[8]]]]> : tensor<1x4x1x1xi8>} : tensor<1x4x1x1x!quant.uniform<i8:f32, 5.000000e-01>>
    %result = "onnx.Add"(%conv, %extra_bias) : (tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<1x4x1x1x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>

    return %result : tensor<1x4x6x6x!quant.uniform<u8:f32, 5.000000e-01:5>>
}
// CHECK: onnx.Conv
// CHECK: onnx.Add
