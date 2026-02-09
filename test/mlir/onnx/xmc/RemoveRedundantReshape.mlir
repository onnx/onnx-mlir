// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// RUN: onnx-mlir-opt --split-input-file --remove-redundant-reshape %s | FileCheck %s

// Test 1: Reshape -> Sigmoid -> Reshape with quantized types
// CHECK-LABEL: @reshape_sigmoid_reshape_quant
func.func @reshape_sigmoid_reshape_quant(%arg0: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>> {
    %shape1 = onnx.Constant dense<[1, 512]> : tensor<2xi64>
    %reshape1 = "onnx.Reshape"(%arg0, %shape1) : (tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<2xi64>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %sigmoid = "onnx.Sigmoid"(%reshape1) : (tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %shape2 = onnx.Constant dense<[1, 16, 32]> : tensor<3xi64>
    %reshape2 = "onnx.Reshape"(%sigmoid, %shape2) : (tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<3xi64>) -> tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
    return %reshape2 : tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
}
// CHECK: onnx.Sigmoid
// CHECK-SAME: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
// CHECK: onnx.Reshape
// First reshape should be eliminated

// -----

// Test 2: Reshape -> Add <- Reshape -> Reshape with quantized types
// CHECK-LABEL: @reshape_add_reshape_quant
func.func @reshape_add_reshape_quant(%arg0: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>, %arg1: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>> {
    %shape1 = onnx.Constant dense<[1, 512]> : tensor<2xi64>
    %reshape1 = "onnx.Reshape"(%arg0, %shape1) : (tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<2xi64>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %reshape2 = "onnx.Reshape"(%arg1, %shape1) : (tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<2xi64>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %add = "onnx.Add"(%reshape1, %reshape2) : (tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %shape2 = onnx.Constant dense<[1, 16, 32]> : tensor<3xi64>
    %reshape3 = "onnx.Reshape"(%add, %shape2) : (tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<3xi64>) -> tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
    return %reshape3 : tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
}
// CHECK: onnx.Add
// CHECK-SAME: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
// CHECK: onnx.Reshape
// Input reshapes should be eliminated

// -----

// Test 3: Reshape -> Mul <- Reshape -> Reshape with quantized types
// CHECK-LABEL: @reshape_mul_reshape_quant
func.func @reshape_mul_reshape_quant(%arg0: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>, %arg1: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>> {
    %shape1 = onnx.Constant dense<[1, 512]> : tensor<2xi64>
    %reshape1 = "onnx.Reshape"(%arg0, %shape1) : (tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<2xi64>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %reshape2 = "onnx.Reshape"(%arg1, %shape1) : (tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<2xi64>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %mul = "onnx.Mul"(%reshape1, %reshape2) : (tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %shape2 = onnx.Constant dense<[1, 16, 32]> : tensor<3xi64>
    %reshape3 = "onnx.Reshape"(%mul, %shape2) : (tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<3xi64>) -> tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
    return %reshape3 : tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
}
// CHECK: onnx.Mul
// CHECK-SAME: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
// CHECK: onnx.Reshape
// Input reshapes should be eliminated

// -----

// Test 4: Reshape -> Sub <- Reshape -> Reshape with quantized types
// CHECK-LABEL: @reshape_sub_reshape_quant
func.func @reshape_sub_reshape_quant(%arg0: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>, %arg1: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>> {
    %shape1 = onnx.Constant dense<[1, 512]> : tensor<2xi64>
    %reshape1 = "onnx.Reshape"(%arg0, %shape1) : (tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<2xi64>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %reshape2 = "onnx.Reshape"(%arg1, %shape1) : (tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<2xi64>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %sub = "onnx.Sub"(%reshape1, %reshape2) : (tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %shape2 = onnx.Constant dense<[1, 16, 32]> : tensor<3xi64>
    %reshape3 = "onnx.Reshape"(%sub, %shape2) : (tensor<1x512x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<3xi64>) -> tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
    return %reshape3 : tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
}
// CHECK: onnx.Sub
// CHECK-SAME: tensor<1x16x32x!quant.uniform<u8:f32, 5.000000e-01:5>>
// CHECK: onnx.Reshape
// Input reshapes should be eliminated
