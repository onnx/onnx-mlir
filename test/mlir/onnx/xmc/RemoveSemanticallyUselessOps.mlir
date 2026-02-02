// RUN: onnx-mlir-opt --split-input-file --remove-semantically-useless-ops %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// CHECK-LABEL: @identity_same_shape
func.func @identity_same_shape(%arg0: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>) {
    %result = "onnx.Identity"(%arg0) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %result : tensor<1x3x4x4xf32>
}
// CHECK-NOT: onnx.Identity
// CHECK: return %arg0

// CHECK-LABEL: @reshape_same_input

func.func @reshape_same_input(%arg0: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>) {
    %shape = onnx.Constant dense<[1, 3, 4, 4]> : tensor<4xi64>
    %result = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} : (tensor<1x3x4x4xf32>, tensor<4xi64>) -> tensor<1x3x4x4xf32>
    return %result : tensor<1x3x4x4xf32>
}
// CHECK-NOT: onnx.Reshape

// CHECK-LABEL: @mul_with_zero_lhs
func.func @mul_with_zero_lhs(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
    %zero = onnx.Constant dense<0.0> : tensor<1x3x4x4xf32>
    %result = "onnx.Mul"(%zero, %arg0) : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %result : tensor<1x3x4x4xf32>
}
// CHECK: %[[ZERO:.*]] = onnx.Constant dense<0.000000e+00> : tensor<1x3x4x4xf32>
// CHECK-NOT: onnx.Mul
// CHECK: return %[[ZERO]]


// CHECK-LABEL: @mul_with_zero_rhs
func.func @mul_with_zero_rhs(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
    %zero = onnx.Constant dense<0.0> : tensor<1x3x4x4xf32>
    %result = "onnx.Mul"(%arg0, %zero) : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %result : tensor<1x3x4x4xf32>
}
// CHECK: %[[ZERO:.*]] = onnx.Constant dense<0.000000e+00> : tensor<1x3x4x4xf32>
// CHECK-NOT: onnx.Mul
// CHECK: return %[[ZERO]]

// CHECK-LABEL: @mul_with_zero_integer
func.func @mul_with_zero_integer(%arg0: tensor<1x3x4x4xi32>) -> tensor<1x3x4x4xi32> {
    %zero = onnx.Constant dense<0> : tensor<1x3x4x4xi32>
    %result = "onnx.Mul"(%arg0, %zero) : (tensor<1x3x4x4xi32>, tensor<1x3x4x4xi32>) -> tensor<1x3x4x4xi32>
    return %result : tensor<1x3x4x4xi32>
}
// CHECK: %[[ZERO:.*]] = onnx.Constant dense<0> : tensor<1x3x4x4xi32>
// CHECK-NOT: onnx.Mul
// CHECK: return %[[ZERO]]

// CHECK-LABEL: @sub_with_same_input
func.func @sub_with_same_input(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
    %result = "onnx.Sub"(%arg0, %arg0) : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %result : tensor<1x3x4x4xf32>
}
// CHECK: %[[ZERO:.*]] = onnx.Constant dense<0.000000e+00> : tensor<1x3x4x4xf32>
// CHECK-NOT: onnx.Sub
// CHECK: return %[[ZERO]]

// CHECK-LABEL: @sub_with_same_input_integer
func.func @sub_with_same_input_integer(%arg0: tensor<1x3x4x4xi32>) -> tensor<1x3x4x4xi32> {
    %result = "onnx.Sub"(%arg0, %arg0) : (tensor<1x3x4x4xi32>, tensor<1x3x4x4xi32>) -> tensor<1x3x4x4xi32>
    return %result : tensor<1x3x4x4xi32>
}
// CHECK: %[[ZERO:.*]] = onnx.Constant dense<0> : tensor<1x3x4x4xi32>
// CHECK-NOT: onnx.Sub
// CHECK: return %[[ZERO]]

// CHECK-LABEL: @resize_with_scale_one
func.func @resize_with_scale_one(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %scales = onnx.Constant dense<1.0> : tensor<4xf32>
    %result = "onnx.Resize"(%arg0, %none, %scales, %none) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<1x3x224x224xf32>, none, tensor<4xf32>, none) -> tensor<1x3x224x224xf32>
    return %result : tensor<1x3x224x224xf32>
}
// CHECK-NOT: onnx.Resize
// CHECK: return %arg0

// CHECK-LABEL: @resize_with_same_shape
func.func @resize_with_same_shape(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %sizes = onnx.Constant dense<[1, 3, 224, 224]> : tensor<4xi64>
    %result = "onnx.Resize"(%arg0, %none, %none, %sizes) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<1x3x224x224xf32>, none, none, tensor<4xi64>) -> tensor<1x3x224x224xf32>
    return %result : tensor<1x3x224x224xf32>
}
// CHECK-NOT: onnx.Resize
// CHECK: return %arg0

// CHECK-LABEL: @pad_with_all_zero_pads
func.func @pad_with_all_zero_pads(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
    %pads = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 0]> : tensor<8xi64>
    %none = "onnx.NoValue"() {value} : () -> none
    %result = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "constant"} : (tensor<1x3x224x224xf32>, tensor<8xi64>, none, none) -> tensor<1x3x224x224xf32>
    return %result : tensor<1x3x224x224xf32>
}
// CHECK-NOT: onnx.Pad
// CHECK: return %arg0
