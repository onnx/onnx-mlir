// RUN: onnx-mlir-opt --split-input-file --convert-instancenorm-to-groupnorm %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// Test case 1: Basic 4D tensor conversion (NCHW format)

// CHECK-LABEL: @convert_instancenorm_to_groupnorm_4d
func.func @convert_instancenorm_to_groupnorm_4d(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32> {
    // Reshape from 1x4x8x8 to 1x2x16x8 (merge channels for InstanceNorm)
    %shape1 = onnx.Constant dense<[1, 2, 16, 8]> : tensor<4xi64>
    %reshaped = "onnx.Reshape"(%arg0, %shape1) {allowzero = 0 : si64} : (tensor<1x4x8x8xf32>, tensor<4xi64>) -> tensor<1x2x16x8xf32>

    // InstanceNorm scale and bias (for 2 channels)
    %scale = onnx.Constant dense<[1.0, 1.0]> : tensor<2xf32>
    %bias = onnx.Constant dense<[0.0, 0.0]> : tensor<2xf32>

    // InstanceNormalization
    %instancenorm = "onnx.InstanceNormalization"(%reshaped, %scale, %bias) {epsilon = 1.0e-5 : f32} : (tensor<1x2x16x8xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x16x8xf32>

    // Reshape back to original shape
    %shape2 = onnx.Constant dense<[1, 4, 8, 8]> : tensor<4xi64>
    %output = "onnx.Reshape"(%instancenorm, %shape2) {allowzero = 0 : si64} : (tensor<1x2x16x8xf32>, tensor<4xi64>) -> tensor<1x4x8x8xf32>

    return %output : tensor<1x4x8x8xf32>
}
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.InstanceNormalization
// CHECK: onnx.GroupNormalization
// CHECK-SAME: num_groups = 2


// Test case 2: 3D tensor conversion (NCD format)

// CHECK-LABEL: @convert_instancenorm_to_groupnorm_3d
func.func @convert_instancenorm_to_groupnorm_3d(%arg0: tensor<1x6x16xf32>) -> tensor<1x6x16xf32> {
    // Reshape from 1x6x16 to 1x3x32
    %shape1 = onnx.Constant dense<[1, 3, 32]> : tensor<3xi64>
    %reshaped = "onnx.Reshape"(%arg0, %shape1) {allowzero = 0 : si64} : (tensor<1x6x16xf32>, tensor<3xi64>) -> tensor<1x3x32xf32>

    // InstanceNorm scale and bias (for 3 channels)
    %scale = onnx.Constant dense<[1.0, 1.0, 1.0]> : tensor<3xf32>
    %bias = onnx.Constant dense<[0.0, 0.0, 0.0]> : tensor<3xf32>

    // InstanceNormalization
    %instancenorm = "onnx.InstanceNormalization"(%reshaped, %scale, %bias) {epsilon = 1.0e-5 : f32} : (tensor<1x3x32xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3x32xf32>

    // Reshape back to original shape
    %shape2 = onnx.Constant dense<[1, 6, 16]> : tensor<3xi64>
    %output = "onnx.Reshape"(%instancenorm, %shape2) {allowzero = 0 : si64} : (tensor<1x3x32xf32>, tensor<3xi64>) -> tensor<1x6x16xf32>

    return %output : tensor<1x6x16xf32>
}
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.InstanceNormalization
// CHECK: onnx.GroupNormalization
// CHECK-SAME: num_groups = 2


// Test case 3: Conversion with non-trivial scale and bias values

// CHECK-LABEL: @convert_instancenorm_to_groupnorm_with_values
func.func @convert_instancenorm_to_groupnorm_with_values(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
    %shape1 = onnx.Constant dense<[1, 4, 8, 4]> : tensor<4xi64>
    %reshaped = "onnx.Reshape"(%arg0, %shape1) {allowzero = 0 : si64} : (tensor<1x8x4x4xf32>, tensor<4xi64>) -> tensor<1x4x8x4xf32>

    // Non-trivial scale and bias values
    %scale = onnx.Constant dense<[0.5, 1.5, 2.0, 0.8]> : tensor<4xf32>
    %bias = onnx.Constant dense<[0.1, 0.2, 0.3, 0.4]> : tensor<4xf32>

    %instancenorm = "onnx.InstanceNormalization"(%reshaped, %scale, %bias) {epsilon = 1.0e-6 : f32} : (tensor<1x4x8x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x8x4xf32>

    %shape2 = onnx.Constant dense<[1, 8, 4, 4]> : tensor<4xi64>
    %output = "onnx.Reshape"(%instancenorm, %shape2) {allowzero = 0 : si64} : (tensor<1x4x8x4xf32>, tensor<4xi64>) -> tensor<1x8x4x4xf32>

    return %output : tensor<1x8x4x4xf32>
}
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.InstanceNormalization
// CHECK: onnx.GroupNormalization
// CHECK-SAME: num_groups = 2


// Test case 4: Negative case - shapes don't match (should not convert)
// Input shape != output shape, so pattern should not match

// CHECK-LABEL: @no_convert_shape_mismatch
func.func @no_convert_shape_mismatch(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x2x16x8xf32> {
    %shape1 = onnx.Constant dense<[1, 2, 16, 8]> : tensor<4xi64>
    %reshaped = "onnx.Reshape"(%arg0, %shape1) {allowzero = 0 : si64} : (tensor<1x4x8x8xf32>, tensor<4xi64>) -> tensor<1x2x16x8xf32>

    %scale = onnx.Constant dense<[1.0, 1.0]> : tensor<2xf32>
    %bias = onnx.Constant dense<[0.0, 0.0]> : tensor<2xf32>

    %instancenorm = "onnx.InstanceNormalization"(%reshaped, %scale, %bias) {epsilon = 1.0e-5 : f32} : (tensor<1x2x16x8xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x16x8xf32>

    // Output shape is different from input shape (no second reshape to original)
    return %instancenorm : tensor<1x2x16x8xf32>
}
// CHECK: onnx.Reshape
// CHECK: onnx.InstanceNormalization
// CHECK-NOT: onnx.GroupNormalization


// Test case 5: Negative case - standalone InstanceNorm without surrounding reshapes

// CHECK-LABEL: @no_convert_standalone_instancenorm
func.func @no_convert_standalone_instancenorm(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32> {
    %scale = onnx.Constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
    %bias = onnx.Constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>

    %instancenorm = "onnx.InstanceNormalization"(%arg0, %scale, %bias) {epsilon = 1.0e-5 : f32} : (tensor<1x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>

    return %instancenorm : tensor<1x4x8x8xf32>
}
// CHECK: onnx.InstanceNormalization
// CHECK-NOT: onnx.GroupNormalization


// Test case 6: Batch size > 1
// Input: 2x4x8x8 (batch=2, 4 channels)

// CHECK-LABEL: @convert_instancenorm_to_groupnorm_batch2
func.func @convert_instancenorm_to_groupnorm_batch2(%arg0: tensor<2x4x8x8xf32>) -> tensor<2x4x8x8xf32> {
    %shape1 = onnx.Constant dense<[2, 2, 16, 8]> : tensor<4xi64>
    %reshaped = "onnx.Reshape"(%arg0, %shape1) {allowzero = 0 : si64} : (tensor<2x4x8x8xf32>, tensor<4xi64>) -> tensor<2x2x16x8xf32>

    %scale = onnx.Constant dense<[1.0, 1.0]> : tensor<2xf32>
    %bias = onnx.Constant dense<[0.0, 0.0]> : tensor<2xf32>

    %instancenorm = "onnx.InstanceNormalization"(%reshaped, %scale, %bias) {epsilon = 1.0e-5 : f32} : (tensor<2x2x16x8xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x2x16x8xf32>

    %shape2 = onnx.Constant dense<[2, 4, 8, 8]> : tensor<4xi64>
    %output = "onnx.Reshape"(%instancenorm, %shape2) {allowzero = 0 : si64} : (tensor<2x2x16x8xf32>, tensor<4xi64>) -> tensor<2x4x8x8xf32>

    return %output : tensor<2x4x8x8xf32>
}
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.InstanceNormalization
// CHECK: onnx.GroupNormalization
// CHECK-SAME: num_groups = 2
