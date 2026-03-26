// RUN: onnx-mlir-opt --fuse-conv-activation %s | FileCheck %s
// NOTE: This pass assumes quant-types, replace-qdq-eltwise, and
// replace-hsigmoid-and-hswish have already run. Activation ops are
// represented as XCOMPILERFusedEltwiseOp with a "type" attribute.

// -----
// Test: XFEConv + LeakyReLU with matching quant params → fused
// CHECK-LABEL: func.func @test_xfeconv_leakyrelu_fusion
func.func @test_xfeconv_leakyrelu_fusion(
    %arg0: tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:128>>,
    %weight: tensor<32x3x3x16x!quant.uniform<i8:f32, 0.01>>,
    %bias: tensor<32x!quant.uniform<i32:f32, 5.000000e-04>>,
    %none: none) -> tensor<1x8x8x32x!quant.uniform<u8:f32, 0.04:128>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:128>>,
         tensor<32x3x3x16x!quant.uniform<i8:f32, 0.01>>,
         tensor<32x!quant.uniform<i32:f32, 5.000000e-04>>)
      -> tensor<1x8x8x32x!quant.uniform<u8:f32, 0.04:128>>

  %act = "onnx.XCOMPILERFusedEltwise"(%conv, %none)
      {enable_lut_sigmoid = false, leakyrelu_alpha = 3.000000e-01 : f32,
       nonlinear = "NONE", prelu_in = 77 : si64, prelu_shift = 8 : si64,
       type = "LEAKYRELU"}
      : (tensor<1x8x8x32x!quant.uniform<u8:f32, 0.04:128>>, none)
      -> tensor<1x8x8x32x!quant.uniform<u8:f32, 0.04:128>>

  return %act : tensor<1x8x8x32x!quant.uniform<u8:f32, 0.04:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "LEAKYRELU"
// CHECK-SAME: leakyrelu_alpha = 3.000000e-01 : f32
// CHECK-NOT: "onnx.XCOMPILERFusedEltwise"

// -----
// Test: No fusion when conv and activation have different quant params
// (requantization detected)
// CHECK-LABEL: func.func @test_no_fusion_different_quant_params
func.func @test_no_fusion_different_quant_params(
    %arg0: tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:128>>,
    %weight: tensor<32x3x3x16x!quant.uniform<i8:f32, 0.01>>,
    %bias: tensor<32x!quant.uniform<i32:f32, 5.000000e-04>>,
    %none: none) -> tensor<1x8x8x32x!quant.uniform<u8:f32, 0.03:100>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:128>>,
         tensor<32x3x3x16x!quant.uniform<i8:f32, 0.01>>,
         tensor<32x!quant.uniform<i32:f32, 5.000000e-04>>)
      -> tensor<1x8x8x32x!quant.uniform<u8:f32, 0.04:128>>

  %act = "onnx.XCOMPILERFusedEltwise"(%conv, %none)
      {enable_lut_sigmoid = false, leakyrelu_alpha = 3.000000e-01 : f32,
       nonlinear = "NONE", prelu_in = 77 : si64, prelu_shift = 8 : si64,
       type = "LEAKYRELU"}
      : (tensor<1x8x8x32x!quant.uniform<u8:f32, 0.04:128>>, none)
      -> tensor<1x8x8x32x!quant.uniform<u8:f32, 0.03:100>>

  return %act : tensor<1x8x8x32x!quant.uniform<u8:f32, 0.03:100>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "NONE"
// CHECK: "onnx.XCOMPILERFusedEltwise"

// -----
// Test: XFEConv + ReLU fusion (same quant params)
// When conv and activation output have the same quant type, no
// conv_output_scale/zero_point is saved.
// CHECK-LABEL: func.func @test_xfeconv_relu_fusion_same_quant
func.func @test_xfeconv_relu_fusion_same_quant(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>,
    %none: none) -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  %act = "onnx.XCOMPILERFusedEltwise"(%conv, %none)
      {enable_lut_sigmoid = false, nonlinear = "NONE", type = "RELU"}
      : (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>, none)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  return %act : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "RELU"
// CHECK-NOT: conv_output_scale
// CHECK-NOT: conv_output_zero_point
// CHECK-NOT: "onnx.XCOMPILERFusedEltwise"

// -----
// Test: XFEConv + QLINEARSIGMOID (quantized hard sigmoid) → activation HSIGMOID
// CHECK-LABEL: func.func @test_xfeconv_qlinearsigmoid_fusion
func.func @test_xfeconv_qlinearsigmoid_fusion(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.05:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.01>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 5.000000e-04>>,
    %none: none) -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.03:128>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.05:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.01>>,
         tensor<16x!quant.uniform<i32:f32, 5.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.03:128>>

  %act = "onnx.XCOMPILERFusedEltwise"(%conv, %none)
      {enable_lut_sigmoid = false, nonlinear = "NONE",
       type = "QLINEARSIGMOID"}
      : (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.03:128>>, none)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.03:128>>

  return %act : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.03:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "HSIGMOID"
// CHECK-NOT: "onnx.XCOMPILERFusedEltwise"

// -----
// Test: No fusion when conv has multiple users
// CHECK-LABEL: func.func @test_no_fusion_multiple_users
func.func @test_no_fusion_multiple_users(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>,
    %none: none) -> (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>,
                     tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>) {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  %act = "onnx.XCOMPILERFusedEltwise"(%conv, %none)
      {enable_lut_sigmoid = false, nonlinear = "NONE", type = "RELU"}
      : (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>, none)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  // Conv has two users: the FusedEltwise AND this return
  return %act, %conv : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>,
                        tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "NONE"
// CHECK: "onnx.XCOMPILERFusedEltwise"

// -----
// Test: No fusion when conv pads attribute contains a negative value
// CHECK-LABEL: func.func @test_no_fusion_negative_pads
func.func @test_no_fusion_negative_pads(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>,
    %none: none) -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>> {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, -1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  %act = "onnx.XCOMPILERFusedEltwise"(%conv, %none)
      {enable_lut_sigmoid = false, nonlinear = "NONE", type = "RELU"}
      : (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>, none)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  return %act : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "NONE"
// CHECK-SAME: pads = [1, 1, 1, -1]
// CHECK: "onnx.XCOMPILERFusedEltwise"

// -----
// Test: No fusion when activation output has multiple users (conv has single user)
// CHECK-LABEL: func.func @test_no_fusion_activation_multi_use
func.func @test_no_fusion_activation_multi_use(
    %arg0: tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
    %weight: tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>,
    %none: none) -> (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>,
                     tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>) {

  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x4x4x8x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<16x3x3x8x!quant.uniform<i8:f32, 0.005>>,
         tensor<16x!quant.uniform<i32:f32, 1.000000e-04>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  %act = "onnx.XCOMPILERFusedEltwise"(%conv, %none)
      {enable_lut_sigmoid = false, nonlinear = "NONE", type = "RELU"}
      : (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>, none)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>

  // Two uses of activation output; conv still has a single user.
  return %act, %act : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>,
                        tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.XFEConv"
// CHECK-SAME: activation = "NONE"
// CHECK: "onnx.XCOMPILERFusedEltwise"

// -----
// Test: No fusion when FusedEltwise is not an activation (e.g., ADD)
// CHECK-LABEL: func.func @test_no_fusion_non_activation
func.func @test_no_fusion_non_activation(
    %arg0: tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>,
    %arg1: tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>)
    -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.04:128>> {

  %add = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1)
      {enable_lut_sigmoid = false, nonlinear = "NONE", type = "ADD"}
      : (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>,
         tensor<1x4x4x16x!quant.uniform<u8:f32, 0.02:128>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.04:128>>

  return %add : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.04:128>>
}
// CHECK: "onnx.XCOMPILERFusedEltwise"
// CHECK-SAME: type = "ADD"

// -----
// Test: XCOMPILERDepthwiseConv + ReLU fusion (matching quant params)
// CHECK-LABEL: func.func @test_depthwiseconv_relu_fusion
func.func @test_depthwiseconv_relu_fusion(
    %arg0: tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:128>>,
    %weight: tensor<1x3x3x16x!quant.uniform<i8:f32, 0.01>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 5.000000e-04>>,
    %none: none) -> tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>> {

  %conv = "onnx.XCOMPILERDepthwiseConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
      : (tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:128>>,
         tensor<1x3x3x16x!quant.uniform<i8:f32, 0.01>>,
         tensor<16x!quant.uniform<i32:f32, 5.000000e-04>>)
      -> tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>>

  %act = "onnx.XCOMPILERFusedEltwise"(%conv, %none)
      {enable_lut_sigmoid = false, nonlinear = "NONE", type = "RELU"}
      : (tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>>, none)
      -> tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>>

  return %act : tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>>
}
// CHECK: "onnx.XCOMPILERDepthwiseConv"
// CHECK-SAME: activation = "RELU"
// CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
