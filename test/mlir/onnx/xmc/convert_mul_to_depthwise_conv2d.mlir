// RUN: onnx-mlir-opt --split-input-file --convert-mul-to-depthwise-conv2d %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// Layout: Dimensions chosen so H=W=C to avoid layout ambiguity with Conv

//===----------------------------------------------------------------------===//
// Mul → DepthwiseConv Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @mul_to_depthwise_conv_4d
// Test basic Mul to DepthwiseConv conversion with 4D input (all dims equal to avoid layout issues)
func.func @mul_to_depthwise_conv_4d(%arg0: tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32> {
    %0 = onnx.Constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = "onnx.Mul"(%arg0, %0) : (tensor<1x3x3x3xf32>, tensor<3xf32>) -> tensor<1x3x3x3xf32>
    return %1 : tensor<1x3x3x3xf32>
}
// CHECK: %[[NONE:.*]] = "onnx.NoValue"()
// CHECK: %[[WEIGHT_CONST:.*]] = onnx.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]>
// CHECK: %[[SHAPE_CONST:.*]] = onnx.Constant dense<[3, 1, 1, 1]>
// CHECK: %[[WEIGHT:.*]] = "onnx.Reshape"(%[[WEIGHT_CONST]], %[[SHAPE_CONST]]) {{.*}} : (tensor<3xf32>, tensor<4xi64>) -> tensor<3x1x1x1xf32>
// CHECK: %[[CONV:.*]] = "onnx.Conv"(%arg0, %[[WEIGHT]], %[[NONE]])
// CHECK-SAME: auto_pad = "NOTSET"
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: group = 3 : si64
// CHECK-SAME: kernel_shape = [1, 1]
// CHECK-SAME: pads = [0, 0, 0, 0]
// CHECK-SAME: strides = [1, 1]
// CHECK-SAME: (tensor<1x3x3x3xf32>, tensor<3x1x1x1xf32>, none) -> tensor<1x3x3x3xf32>
// CHECK: return %[[CONV]]
// CHECK-NOT: onnx.Mul

// CHECK-LABEL: @mul_to_depthwise_conv_weight_first
// Test with weight as first operand (commutative)
func.func @mul_to_depthwise_conv_weight_first(%arg0: tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32> {
    %0 = onnx.Constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = "onnx.Mul"(%0, %arg0) : (tensor<3xf32>, tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    return %1 : tensor<1x3x3x3xf32>
}
// CHECK: "onnx.NoValue"()
// CHECK: onnx.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]>
// CHECK: onnx.Constant dense<[3, 1, 1, 1]>
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Conv"
// CHECK-SAME: group = 3 : si64
// CHECK-NOT: onnx.Mul

// CHECK-LABEL: @mul_to_depthwise_conv_larger_channels
// Test with larger number of channels (16x16x16)
func.func @mul_to_depthwise_conv_larger_channels(%arg0: tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32> {
    %0 = onnx.Constant dense<2.0> : tensor<16xf32>
    %1 = "onnx.Mul"(%arg0, %0) : (tensor<1x16x16x16xf32>, tensor<16xf32>) -> tensor<1x16x16x16xf32>
    return %1 : tensor<1x16x16x16xf32>
}
// CHECK: "onnx.Reshape"{{.*}}-> tensor<16x1x1x1xf32>
// CHECK: "onnx.Conv"
// CHECK-SAME: group = 16 : si64
// CHECK-NOT: onnx.Mul

//===----------------------------------------------------------------------===//
// Mul+Add → DepthwiseConv (with bias) Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @mul_add_to_depthwise_conv_with_bias
// Test Mul+Add fusion to DepthwiseConv with bias
func.func @mul_add_to_depthwise_conv_with_bias(%arg0: tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32> {
    %weight = onnx.Constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %bias = onnx.Constant dense<[0.1, 0.2, 0.3]> : tensor<3xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<1x3x3x3xf32>, tensor<3xf32>) -> tensor<1x3x3x3xf32>
    %add = "onnx.Add"(%mul, %bias) : (tensor<1x3x3x3xf32>, tensor<3xf32>) -> tensor<1x3x3x3xf32>
    return %add : tensor<1x3x3x3xf32>
}
// CHECK: %[[WEIGHT:.*]] = onnx.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]>
// CHECK: %[[BIAS:.*]] = onnx.Constant dense<[1.000000e-01, 2.000000e-01, 3.000000e-01]>
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Conv"({{.*}}, {{.*}}, %[[BIAS]])
// CHECK-SAME: group = 3 : si64
// CHECK-NOT: onnx.Mul
// CHECK-NOT: onnx.Add

// CHECK-LABEL: @mul_add_to_depthwise_conv_bias_first
// Test with bias on the left side of Add
func.func @mul_add_to_depthwise_conv_bias_first(%arg0: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
    %weight = onnx.Constant dense<1.5> : tensor<8xf32>
    %bias = onnx.Constant dense<0.5> : tensor<8xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<1x8x8x8xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
    %add = "onnx.Add"(%bias, %mul) : (tensor<8xf32>, tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
    return %add : tensor<1x8x8x8xf32>
}
// CHECK: onnx.Constant dense<1.500000e+00>
// CHECK: onnx.Constant dense<5.000000e-01>
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Conv"
// CHECK-SAME: group = 8 : si64
// CHECK-NOT: onnx.Mul
// CHECK-NOT: onnx.Add

//===----------------------------------------------------------------------===//
// Mul+Relu → DepthwiseConv+Relu Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @mul_relu_to_depthwise_conv_relu
// Test Mul+Relu fusion to DepthwiseConv+Relu
func.func @mul_relu_to_depthwise_conv_relu(%arg0: tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32> {
    %weight = onnx.Constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<1x3x3x3xf32>, tensor<3xf32>) -> tensor<1x3x3x3xf32>
    %relu = "onnx.Relu"(%mul) : (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    return %relu : tensor<1x3x3x3xf32>
}
// CHECK: "onnx.NoValue"()
// CHECK: onnx.Constant
// CHECK: onnx.Constant
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Conv"
// CHECK-SAME: group = 3 : si64
// CHECK: "onnx.Relu"
// CHECK-NOT: onnx.Mul

// CHECK-LABEL: @mul_relu_to_depthwise_conv_relu_larger
// Test with larger tensors (16x16x16)
func.func @mul_relu_to_depthwise_conv_relu_larger(%arg0: tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32> {
    %weight = onnx.Constant dense<2.0> : tensor<16xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<1x16x16x16xf32>, tensor<16xf32>) -> tensor<1x16x16x16xf32>
    %relu = "onnx.Relu"(%mul) : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    return %relu : tensor<1x16x16x16xf32>
}
// CHECK: "onnx.Conv"
// CHECK-SAME: group = 16 : si64
// CHECK: "onnx.Relu"
// CHECK-NOT: onnx.Mul

//===----------------------------------------------------------------------===//
// Negative Tests - Should NOT be transformed
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @mul_no_constant_no_transform
// Test that Mul without constant is not transformed
func.func @mul_no_constant_no_transform(%arg0: tensor<1x3x3x3xf32>, %arg1: tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32> {
    %mul = "onnx.Mul"(%arg0, %arg1) : (tensor<1x3x3x3xf32>, tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    return %mul : tensor<1x3x3x3xf32>
}
// CHECK: "onnx.Mul"
// CHECK-NOT: onnx.Conv

// CHECK-LABEL: @mul_5d_input_no_transform
// Test that Mul with 5D input is not transformed (pass only handles <= 4D)
func.func @mul_5d_input_no_transform(%arg0: tensor<1x2x4x4x8xf32>) -> tensor<1x2x4x4x8xf32> {
    %weight = onnx.Constant dense<2.0> : tensor<8xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<1x2x4x4x8xf32>, tensor<8xf32>) -> tensor<1x2x4x4x8xf32>
    return %mul : tensor<1x2x4x4x8xf32>
}
// CHECK: "onnx.Mul"
// CHECK-NOT: onnx.Conv

// CHECK-LABEL: @mul_add_non_constant_bias_no_transform
// Test that Mul+Add with non-constant bias is not transformed
func.func @mul_add_non_constant_bias_no_transform(%arg0: tensor<1x3x3x3xf32>, %arg1: tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32> {
    %weight = onnx.Constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<1x3x3x3xf32>, tensor<3xf32>) -> tensor<1x3x3x3xf32>
    %add = "onnx.Add"(%mul, %arg1) : (tensor<1x3x3x3xf32>, tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    return %add : tensor<1x3x3x3xf32>
}
// Since the bias is not constant, the pattern should not match at all
// CHECK-NOT: "onnx.Conv"
// CHECK: "onnx.Add"

// CHECK-LABEL: @mul_multiple_uses_no_transform
// Test that Mul with multiple uses is not transformed
func.func @mul_multiple_uses_no_transform(%arg0: tensor<1x3x3x3xf32>) -> (tensor<1x3x3x3xf32>, tensor<1x3x3x3xf32>) {
    %weight = onnx.Constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<1x3x3x3xf32>, tensor<3xf32>) -> tensor<1x3x3x3xf32>
    %add = "onnx.Add"(%mul, %mul) : (tensor<1x3x3x3xf32>, tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    return %mul, %add : tensor<1x3x3x3xf32>, tensor<1x3x3x3xf32>
}
// CHECK: "onnx.Mul"
// CHECK-NOT: onnx.Conv

//===----------------------------------------------------------------------===//
// Edge Cases and Different Tensor Shapes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @mul_to_depthwise_conv_single_channel
// Single channel depthwise conv (1x1x1)
func.func @mul_to_depthwise_conv_single_channel(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32> {
    %weight = onnx.Constant dense<[2.0]> : tensor<1xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x1x1xf32>
    return %mul : tensor<1x1x1x1xf32>
}
// CHECK: "onnx.Reshape"{{.*}}-> tensor<1x1x1x1xf32>
// CHECK: "onnx.Conv"
// CHECK-SAME: group = 1 : si64
// CHECK-NOT: onnx.Mul

// CHECK-LABEL: @mul_to_depthwise_conv_batch_size_gt_1
// Batch size > 1 should still work (8 batch, 3x3x3)
func.func @mul_to_depthwise_conv_batch_size_gt_1(%arg0: tensor<8x3x3x3xf32>) -> tensor<8x3x3x3xf32> {
    %weight = onnx.Constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<8x3x3x3xf32>, tensor<3xf32>) -> tensor<8x3x3x3xf32>
    return %mul : tensor<8x3x3x3xf32>
}
// CHECK: "onnx.Conv"
// CHECK-SAME: group = 3 : si64
// CHECK-NOT: onnx.Mul

// CHECK-LABEL: @mul_to_depthwise_conv_larger
// Larger dimensions (8x8x8)
func.func @mul_to_depthwise_conv_larger(%arg0: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
    %weight = onnx.Constant dense<1.5> : tensor<8xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<1x8x8x8xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
    return %mul : tensor<1x8x8x8xf32>
}
// CHECK: "onnx.Conv"
// CHECK-SAME: group = 8 : si64
// CHECK-NOT: onnx.Mul

// CHECK-LABEL: @mul_add_relu_chain
// Test chaining: Mul → Add → Relu (Mul+Add pattern should apply)
func.func @mul_add_relu_chain(%arg0: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
    %weight = onnx.Constant dense<1.5> : tensor<8xf32>
    %bias = onnx.Constant dense<0.5> : tensor<8xf32>
    %mul = "onnx.Mul"(%arg0, %weight) : (tensor<1x8x8x8xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
    %add = "onnx.Add"(%mul, %bias) : (tensor<1x8x8x8xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
    %relu = "onnx.Relu"(%add) : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
    return %relu : tensor<1x8x8x8xf32>
}
// CHECK-DAG: onnx.Constant dense<[8, 1, 1, 1]>
// CHECK-DAG: onnx.Constant dense<1.500000e+00>
// CHECK-DAG: onnx.Constant dense<5.000000e-01>
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Conv"
// CHECK: "onnx.Relu"
// CHECK-NOT: onnx.Mul
// CHECK-NOT: onnx.Add
