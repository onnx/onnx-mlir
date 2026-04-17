// RUN: onnx-mlir-opt --normalize-conv-activation %s | FileCheck %s
// NOTE: Runs after fuse-conv-activation. ReLU (α=0) stays "RELU" unless
// implicit in UINT8 zp=0 → NONE; non-native LEAKYRELU α maps to PRELU+M/N.

// -----
// Test: RELU on UINT8 output with zero_point=0 → NONE (implicit ReLU)
// CHECK-LABEL: func.func @test_relu_implicit_uint8_zp0
func.func @test_relu_implicit_uint8_zp0(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
    -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:0>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "RELU", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:0>>

  return %conv : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:0>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "NONE"
// CHECK-NOT: prelu_in
// CHECK-NOT: prelu_shift

// -----
// Test: RELU on UINT8 output with zero_point!=0 → stays RELU (not lowered to PRELU)
// CHECK-LABEL: func.func @test_relu_uint8_nonzero_zp_stays_relu
func.func @test_relu_uint8_nonzero_zp_stays_relu(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
    -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "RELU", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  return %conv : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "RELU"
// CHECK-NOT: prelu_in
// CHECK-NOT: prelu_shift

// -----
// Test: RELU on signed INT8 output → stays RELU (not lowered to PRELU)
// CHECK-LABEL: func.func @test_relu_signed_stays_relu
func.func @test_relu_signed_stays_relu(
    %arg0: tensor<1x4x4x8x!quant.uniform<i8:f32, 0.02:0>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
    -> tensor<1x4x4x16x!quant.uniform<i8:f32, 0.02:0>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "RELU", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<i8:f32, 0.02:0>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<i8:f32, 0.02:0>>

  return %conv : tensor<1x4x4x16x!quant.uniform<i8:f32, 0.02:0>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "RELU"
// CHECK-NOT: prelu_in
// CHECK-NOT: prelu_shift

// -----
// Test: LEAKYRELU with explicit alpha=0 → RELU (same as plain ReLU)
// CHECK-LABEL: func.func @test_leakyrelu_alpha_zero_to_relu
func.func @test_leakyrelu_alpha_zero_to_relu(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
    -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "LEAKYRELU", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3],
       leakyrelu_alpha = 0.000000e+00 : f32,
       pads = [1, 1, 1, 1], strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  return %conv : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "RELU"
// CHECK-NOT: prelu_in
// CHECK-NOT: prelu_shift

// -----
// Test: LEAKYRELU with alpha=26/256 stays LEAKYRELU (hardware-native)
// CHECK-LABEL: func.func @test_leakyrelu_standard_alpha
func.func @test_leakyrelu_standard_alpha(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
    -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>> {

  // 26/256 = 0.1015625
  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "LEAKYRELU", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3],
       leakyrelu_alpha = 0.1015625 : f32,
       pads = [1, 1, 1, 1], strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  return %conv : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "LEAKYRELU"
// CHECK-NOT: activation = "PRELU"
// CHECK-NOT: prelu_in
// CHECK-NOT: prelu_shift

// -----
// Test: LEAKYRELU with alpha!=26/256 → PRELU with mul/shift
// CHECK-LABEL: func.func @test_leakyrelu_nonstandard_alpha
func.func @test_leakyrelu_nonstandard_alpha(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
    -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>> {

  // alpha=0.3 != 26/256 → should become PRELU
  // M = round(2^8 * 0.3) = round(76.8) = 77, N = 8
  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "LEAKYRELU", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3],
       leakyrelu_alpha = 3.000000e-01 : f32,
       pads = [1, 1, 1, 1], strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  return %conv : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "PRELU"
// CHECK-SAME: prelu_in = 77 : si64
// CHECK-SAME: prelu_shift = 8 : si64

// -----
// Test: HSIGMOID passes through unchanged (name used after fuse-conv-activation)
// CHECK-LABEL: func.func @test_hsigmoid_passthrough
func.func @test_hsigmoid_passthrough(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.05:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.01>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 5.000000e-04>>)
    -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.004:0>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "HSIGMOID", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.05:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.01>>,
         tensor<16x!quant.uniform<i32:f32, 5.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.004:0>>

  return %conv : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.004:0>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "HSIGMOID"

// -----
// Test: RELU6 passes through unchanged
// CHECK-LABEL: func.func @test_relu6_passthrough
func.func @test_relu6_passthrough(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
    -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "RELU6", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  return %conv : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "RELU6"

// -----
// Test: NONE activation is not modified
// CHECK-LABEL: func.func @test_none_unchanged
func.func @test_none_unchanged(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
    -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  return %conv : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "NONE"
// CHECK-NOT: prelu_in
// CHECK-NOT: prelu_shift
