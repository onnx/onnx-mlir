// RUN: onnx-mlir-opt --split-input-file --convert-instancenorm-to-groupnorm %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// Test case 1: Basic 4D tensor conversion (NCHW format)

// CHECK-LABEL: @convert_instancenorm_to_groupnorm_4d
func.func @convert_instancenorm_to_groupnorm_4d(%arg0: tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>> {
    %shape1 = onnx.Constant dense<[1, 2, 16, 8]> : tensor<4xi64>
    %reshaped = "onnx.Reshape"(%arg0, %shape1) {allowzero = 0 : si64} : (tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<4xi64>) -> tensor<1x2x16x8x!quant.uniform<i8:f32, 0.05:0>>

    %scale = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi8>} : () -> tensor<2x!quant.uniform<i8:f32, 0.05:0>>
    %bias = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi8>} : () -> tensor<2x!quant.uniform<i8:f32, 0.05:0>>

    %instancenorm = "onnx.InstanceNormalization"(%reshaped, %scale, %bias) {epsilon = 1.0e-5 : f32} : (tensor<1x2x16x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<2x!quant.uniform<i8:f32, 0.05:0>>, tensor<2x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x2x16x8x!quant.uniform<i8:f32, 0.05:0>>

    %shape2 = onnx.Constant dense<[1, 4, 8, 8]> : tensor<4xi64>
    %output = "onnx.Reshape"(%instancenorm, %shape2) {allowzero = 0 : si64} : (tensor<1x2x16x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<4xi64>) -> tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>>

    return %output : tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.InstanceNormalization
// CHECK: onnx.GroupNormalization
// CHECK-SAME: num_groups = 2


// Test case 2: 3D tensor conversion (NCD format)

// CHECK-LABEL: @convert_instancenorm_to_groupnorm_3d
func.func @convert_instancenorm_to_groupnorm_3d(%arg0: tensor<1x6x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x6x16x!quant.uniform<i8:f32, 0.05:0>> {
    %shape1 = onnx.Constant dense<[1, 3, 32]> : tensor<3xi64>
    %reshaped = "onnx.Reshape"(%arg0, %shape1) {allowzero = 0 : si64} : (tensor<1x6x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<3xi64>) -> tensor<1x3x32x!quant.uniform<i8:f32, 0.05:0>>

    %scale = "onnx.Constant"() {value = dense<[1, 1, 1]> : tensor<3xi8>} : () -> tensor<3x!quant.uniform<i8:f32, 0.05:0>>
    %bias = "onnx.Constant"() {value = dense<[0, 0, 0]> : tensor<3xi8>} : () -> tensor<3x!quant.uniform<i8:f32, 0.05:0>>

    %instancenorm = "onnx.InstanceNormalization"(%reshaped, %scale, %bias) {epsilon = 1.0e-5 : f32} : (tensor<1x3x32x!quant.uniform<i8:f32, 0.05:0>>, tensor<3x!quant.uniform<i8:f32, 0.05:0>>, tensor<3x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x32x!quant.uniform<i8:f32, 0.05:0>>

    %shape2 = onnx.Constant dense<[1, 6, 16]> : tensor<3xi64>
    %output = "onnx.Reshape"(%instancenorm, %shape2) {allowzero = 0 : si64} : (tensor<1x3x32x!quant.uniform<i8:f32, 0.05:0>>, tensor<3xi64>) -> tensor<1x6x16x!quant.uniform<i8:f32, 0.05:0>>

    return %output : tensor<1x6x16x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.InstanceNormalization
// CHECK: onnx.GroupNormalization
// CHECK-SAME: num_groups = 2


// Test case 3: Conversion with non-trivial scale and bias values

// CHECK-LABEL: @convert_instancenorm_to_groupnorm_with_values
func.func @convert_instancenorm_to_groupnorm_with_values(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %shape1 = onnx.Constant dense<[1, 4, 8, 4]> : tensor<4xi64>
    %reshaped = "onnx.Reshape"(%arg0, %shape1) {allowzero = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<4xi64>) -> tensor<1x4x8x4x!quant.uniform<i8:f32, 0.05:0>>

    %scale = "onnx.Constant"() {value = dense<[10, 30, 40, 16]> : tensor<4xi8>} : () -> tensor<4x!quant.uniform<i8:f32, 0.05:0>>
    %bias = "onnx.Constant"() {value = dense<[2, 4, 6, 8]> : tensor<4xi8>} : () -> tensor<4x!quant.uniform<i8:f32, 0.05:0>>

    %instancenorm = "onnx.InstanceNormalization"(%reshaped, %scale, %bias) {epsilon = 1.0e-6 : f32} : (tensor<1x4x8x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<4x!quant.uniform<i8:f32, 0.05:0>>, tensor<4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x8x4x!quant.uniform<i8:f32, 0.05:0>>

    %shape2 = onnx.Constant dense<[1, 8, 4, 4]> : tensor<4xi64>
    %output = "onnx.Reshape"(%instancenorm, %shape2) {allowzero = 0 : si64} : (tensor<1x4x8x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<4xi64>) -> tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>

    return %output : tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.InstanceNormalization
// CHECK: onnx.GroupNormalization
// CHECK-SAME: num_groups = 2


// Test case 4: Negative case - shapes don't match (should not convert)

// CHECK-LABEL: @no_convert_shape_mismatch
func.func @no_convert_shape_mismatch(%arg0: tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x2x16x8x!quant.uniform<i8:f32, 0.05:0>> {
    %shape1 = onnx.Constant dense<[1, 2, 16, 8]> : tensor<4xi64>
    %reshaped = "onnx.Reshape"(%arg0, %shape1) {allowzero = 0 : si64} : (tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<4xi64>) -> tensor<1x2x16x8x!quant.uniform<i8:f32, 0.05:0>>

    %scale = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi8>} : () -> tensor<2x!quant.uniform<i8:f32, 0.05:0>>
    %bias = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi8>} : () -> tensor<2x!quant.uniform<i8:f32, 0.05:0>>

    %instancenorm = "onnx.InstanceNormalization"(%reshaped, %scale, %bias) {epsilon = 1.0e-5 : f32} : (tensor<1x2x16x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<2x!quant.uniform<i8:f32, 0.05:0>>, tensor<2x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x2x16x8x!quant.uniform<i8:f32, 0.05:0>>

    return %instancenorm : tensor<1x2x16x8x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: onnx.Reshape
// CHECK: onnx.InstanceNormalization
// CHECK-NOT: onnx.GroupNormalization


// Test case 5: Negative case - standalone InstanceNorm without surrounding reshapes

// CHECK-LABEL: @no_convert_standalone_instancenorm
func.func @no_convert_standalone_instancenorm(%arg0: tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>> {
    %scale = "onnx.Constant"() {value = dense<[1, 1, 1, 1]> : tensor<4xi8>} : () -> tensor<4x!quant.uniform<i8:f32, 0.05:0>>
    %bias = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi8>} : () -> tensor<4x!quant.uniform<i8:f32, 0.05:0>>

    %instancenorm = "onnx.InstanceNormalization"(%arg0, %scale, %bias) {epsilon = 1.0e-5 : f32} : (tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<4x!quant.uniform<i8:f32, 0.05:0>>, tensor<4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>>

    return %instancenorm : tensor<1x4x8x8x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: onnx.InstanceNormalization
// CHECK-NOT: onnx.GroupNormalization


// Test case 6: Batch size > 1

// CHECK-LABEL: @convert_instancenorm_to_groupnorm_batch2
func.func @convert_instancenorm_to_groupnorm_batch2(%arg0: tensor<2x4x8x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<2x4x8x8x!quant.uniform<i8:f32, 0.05:0>> {
    %shape1 = onnx.Constant dense<[2, 2, 16, 8]> : tensor<4xi64>
    %reshaped = "onnx.Reshape"(%arg0, %shape1) {allowzero = 0 : si64} : (tensor<2x4x8x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<4xi64>) -> tensor<2x2x16x8x!quant.uniform<i8:f32, 0.05:0>>

    %scale = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi8>} : () -> tensor<2x!quant.uniform<i8:f32, 0.05:0>>
    %bias = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi8>} : () -> tensor<2x!quant.uniform<i8:f32, 0.05:0>>

    %instancenorm = "onnx.InstanceNormalization"(%reshaped, %scale, %bias) {epsilon = 1.0e-5 : f32} : (tensor<2x2x16x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<2x!quant.uniform<i8:f32, 0.05:0>>, tensor<2x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<2x2x16x8x!quant.uniform<i8:f32, 0.05:0>>

    %shape2 = onnx.Constant dense<[2, 4, 8, 8]> : tensor<4xi64>
    %output = "onnx.Reshape"(%instancenorm, %shape2) {allowzero = 0 : si64} : (tensor<2x2x16x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<4xi64>) -> tensor<2x4x8x8x!quant.uniform<i8:f32, 0.05:0>>

    return %output : tensor<2x4x8x8x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.InstanceNormalization
// CHECK: onnx.GroupNormalization
// CHECK-SAME: num_groups = 2

// Quantized constant with scast

// CHECK-LABEL: @const_scast
func.func @const_scast(%arg0: tensor<1x128x512x512xui16>) -> tensor<1x128x512x512xui16> {
  %0 = onnx.Constant dense<[0, 32, -1]> : tensor<3xi64>
  %1 = onnx.Constant dense<[1, 128, 512, 512]> : tensor<4xi64>
  %2 = onnx.Constant {value = dense<255> : tensor<32xui8>} : tensor<32x!quant.uniform<u8:f32, 0.0039215688593685627>>
  %3 = onnx.Constant dense<0> : tensor<32xi32>
  %4 = quant.scast %3 : tensor<32xi32> to tensor<32x!quant.uniform<i32:f32, 0.029561372473835945>>
  %5 = quant.scast %arg0 : tensor<1x128x512x512xui16> to tensor<1x128x512x512x!quant.uniform<u16:f32, 7.538149356842041:54325>>
  %6 = "onnx.Reshape"(%5, %0) {allowzero = 0 : si64, onnx_node_name = "/decoder/conv_norm_out/Reshape"} : (tensor<1x128x512x512x!quant.uniform<u16:f32, 7.538149356842041:54325>>, tensor<3xi64>) -> tensor<1x32x1048576x!quant.uniform<u16:f32, 7.538149356842041:54325>>
  %7 = "onnx.InstanceNormalization"(%6, %2, %4) {epsilon = 9.99999997E-7 : f32, onnx_node_name = "/decoder/conv_norm_out/InstanceNormalization"} : (tensor<1x32x1048576x!quant.uniform<u16:f32, 7.538149356842041:54325>>, tensor<32x!quant.uniform<u8:f32, 0.0039215688593685627>>, tensor<32x!quant.uniform<i32:f32, 0.029561372473835945>>) -> tensor<1x32x1048576x!quant.uniform<u16:f32, 0.0026107656303793192:58320>>
  %8 = "onnx.Reshape"(%7, %1) {allowzero = 0 : si64, onnx_node_name = "/decoder/conv_norm_out/Reshape_1"} : (tensor<1x32x1048576x!quant.uniform<u16:f32, 0.0026107656303793192:58320>>, tensor<4xi64>) -> tensor<1x128x512x512x!quant.uniform<u16:f32, 0.0026107656303793192:58320>>
  %9 = quant.scast %8 : tensor<1x128x512x512x!quant.uniform<u16:f32, 0.0026107656303793192:58320>> to tensor<1x128x512x512xui16>
  return %9 : tensor<1x128x512x512xui16>
}

// CHECK: onnx.GroupNormalization
// CHECK-NOT: onnx.InstanceNormalization
