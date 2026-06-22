// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Shape inference tests for XCOMPILER Operations
/// Domain: com.amd.xcompiler
//===----------------------------------------------------------------------===//

// -----

//===----------------------------------------------------------------------===//
/// XCOMPILER FusedEltwise Tests (Quantized Element-wise Operations)
//===----------------------------------------------------------------------===//

// COM: Test basic element-wise add with same shapes (no broadcast needed)
func.func @test_XCOMPILER_fused_eltwise_same_shape(%arg0: tensor<1x64x28x28xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "ADD",
    nonlinear = "NONE"
  } : (tensor<1x64x28x28xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_same_shape
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test broadcasting with scalar-like tensor [1] x [N,C,H,W]
func.func @test_XCOMPILER_fused_eltwise_broadcast_scalar(%arg0: tensor<1xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "MUL",
    nonlinear = "NONE"
  } : (tensor<1xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_broadcast_scalar
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test broadcasting with channel dimension [1,C,1,1] x [N,C,H,W]
func.func @test_XCOMPILER_fused_eltwise_broadcast_channel(%arg0: tensor<1x64x1x1xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "ADD",
    nonlinear = "RELU"
  } : (tensor<1x64x1x1xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_broadcast_channel
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test broadcasting with different ranks [1,C,1,1] x [N,C,H,W] -> channel-wise broadcast
func.func @test_XCOMPILER_fused_eltwise_broadcast_rank_diff(%arg0: tensor<1x64x1x1xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "SUB",
    nonlinear = "NONE"
  } : (tensor<1x64x1x1xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_broadcast_rank_diff
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test broadcasting [1,4,1] x [3,1,5] -> [3,4,5]
func.func @test_XCOMPILER_fused_eltwise_broadcast_numpy(%arg0: tensor<1x4x1xi8>, %arg1: tensor<3x1x5xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "DIV",
    nonlinear = "NONE"
  } : (tensor<1x4x1xi8>, tensor<3x1x5xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_broadcast_numpy
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<3x4x5xi8>
  // CHECK: onnx.Return [[RES]] : tensor<3x4x5xi8>
}

// -----

// COM: Test with LeakyReLU activation
func.func @test_XCOMPILER_fused_eltwise_leaky_relu(%arg0: tensor<1x64x28x28xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "ADD",
    nonlinear = "LEAKYRELU",
    leakyrelu_alpha = 0.01 : f32,
    prelu_in = 2621 : si64,
    prelu_shift = 18 : si64
  } : (tensor<1x64x28x28xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_leaky_relu
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test with dynamic dimensions - broadcasting resolves known dims
func.func @test_XCOMPILER_fused_eltwise_dynamic(%arg0: tensor<?x64x?x?xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "ADD",
    nonlinear = "NONE"
  } : (tensor<?x64x?x?xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_dynamic
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<?x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<?x64x28x28xi8>
}

// -----

// COM: Test with single input (B is optional, using none)
func.func @test_XCOMPILER_fused_eltwise_single_input(%arg0: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %none) {
    type = "ADD",
    nonlinear = "RELU"
  } : (tensor<1x64x28x28xi8>, none) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_single_input
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %0) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test that shape inference preserves existing quantized element type
// COM: When the result already has a ranked type with its own scale/zero_point,
// COM: shape inference must NOT overwrite it with input A's element type.
func.func @test_XCOMPILER_fused_eltwise_preserve_quant_type(
    %arg0: tensor<1x512x512x128x!quant.uniform<u16:f32, 3.4006399801000953E-4:29409>>,
    %arg1: tensor<1x512x512x128x!quant.uniform<u16:f32, 1.52587890625E-5>>) ->
    tensor<1x512x512x128x!quant.uniform<u16:f32, 1.9170573796145618E-4:1453>> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "MUL",
    nonlinear = "NONE"
  } : (tensor<1x512x512x128x!quant.uniform<u16:f32, 3.4006399801000953E-4:29409>>,
       tensor<1x512x512x128x!quant.uniform<u16:f32, 1.52587890625E-5>>) ->
       tensor<1x512x512x128x!quant.uniform<u16:f32, 1.9170573796145618E-4:1453>>
  onnx.Return %0 : tensor<1x512x512x128x!quant.uniform<u16:f32, 1.9170573796145618E-4:1453>>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_preserve_quant_type
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x512x512x128x!quant.uniform<u16:f32, 1.9170573796145618E-4:1453>>
  // CHECK: onnx.Return [[RES]] : tensor<1x512x512x128x!quant.uniform<u16:f32, 1.9170573796145618E-4:1453>>
}

// -----

// COM: Test that an unranked result with a quantized element type keeps that
// COM: element type verbatim after shape inference — only the shape is
// COM: refined to match the broadcasted operand shape.
func.func @test_XCOMPILER_fused_eltwise_unranked_quant(
    %arg0: tensor<1x64x28x28x!quant.uniform<u16:f32, 0.001:100>>,
    %arg1: tensor<1x64x28x28x!quant.uniform<u16:f32, 0.002:200>>) ->
    tensor<*x!quant.uniform<u16:f32, 0.001:100>> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "ADD",
    nonlinear = "NONE"
  } : (tensor<1x64x28x28x!quant.uniform<u16:f32, 0.001:100>>,
       tensor<1x64x28x28x!quant.uniform<u16:f32, 0.002:200>>) ->
       tensor<*x!quant.uniform<u16:f32, 0.001:100>>
  onnx.Return %0 : tensor<*x!quant.uniform<u16:f32, 0.001:100>>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_unranked_quant
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28x!quant.uniform<u16:f32, 1.{{.*}}:100>>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28x!quant.uniform<u16:f32, 1.{{.*}}:100>>
}

// -----

// COM: Test that an f32 result is PRESERVED — shape inference infers shape
// COM: only and must not overwrite the result element type with operand A's
// COM: quantized type, even when both operands are quantized. Only the shape
// COM: should be refined; the f32 element type stays.
func.func @test_XCOMPILER_fused_eltwise_f32_result_preserved(
    %arg0: tensor<1x300x1x!quant.uniform<u8:f32, 0.1:128>>,
    %arg1: tensor<1x300x1x!quant.uniform<u8:f32, 0.2:64>>) ->
    tensor<1x300x1xf32> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "ADD",
    nonlinear = "NONE"
  } : (tensor<1x300x1x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x300x1x!quant.uniform<u8:f32, 0.2:64>>) ->
       tensor<1x300x1xf32>
  onnx.Return %0 : tensor<1x300x1xf32>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_f32_result_preserved
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x300x1xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x300x1xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XCOMPILER DepthwiseConv Tests (Depthwise Separable Convolution - NHWC layout)
/// Supports both 2D (4D tensors) and 3D (5D tensors)
/// Weight format: IHWO [1, kH, kW, C] for 2D, [1, kD, kH, kW, C] for 3D
/// (transposed from OHWI by ConvertXFEConvToDepthwiseConvPass)
//===----------------------------------------------------------------------===//

// COM: Test basic 2D depthwise conv with 3x3 kernel, no padding (NHWC layout)
// Input: [N=1, H=28, W=28, C=64], Weight IHWO: [M=1, kH=3, kW=3, C=64]
// Output: [N=1, H=26, W=26, C=64]
func.func @test_XCOMPILER_depthwise_conv_basic(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<1x3x3x64xi8>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<1x3x3x64xi8>, none) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_depthwise_conv_basic
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %0) {{.*}} -> tensor<1x26x26x64xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x26x26x64xi8>
}

// -----

// COM: Test 2D depthwise conv with explicit padding (NHWC layout)
// Input: [N=1, H=14, W=14, C=32], Weight IHWO: [M=1, kH=3, kW=3, C=32]
// Pads: [1,1,1,1] -> Output: [N=1, H=14, W=14, C=32] (same size)
func.func @test_XCOMPILER_depthwise_conv_with_padding(%arg0: tensor<1x14x14x32xi8>, %arg1: tensor<1x3x3x32xi8>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    pads = [1, 1, 1, 1],
    auto_pad = "NOTSET"
  } : (tensor<1x14x14x32xi8>, tensor<1x3x3x32xi8>, none) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_depthwise_conv_with_padding
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %0) {{.*}} -> tensor<1x14x14x32xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x14x14x32xi8>
}

// -----

// COM: Test 2D depthwise conv with stride 2 (NHWC layout)
// Input: [N=1, H=28, W=28, C=64], Weight IHWO: [M=1, kH=3, kW=3, C=64]
// Stride: [2,2], Pads: [1,1,1,1] -> Output: [N=1, H=14, W=14, C=64]
func.func @test_XCOMPILER_depthwise_conv_stride2(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<1x3x3x64xi8>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    strides = [2, 2],
    pads = [1, 1, 1, 1],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<1x3x3x64xi8>, none) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_depthwise_conv_stride2
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %0) {{.*}} -> tensor<1x14x14x64xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x14x14x64xi8>
}

// -----

// COM: Test 2D depthwise conv with bias (NHWC layout)
// Input: [N=1, H=32, W=32, C=16], Weight IHWO: [M=1, kH=5, kW=5, C=16], Bias: [16]
// Output: [N=1, H=28, W=28, C=16]
func.func @test_XCOMPILER_depthwise_conv_with_bias(%arg0: tensor<1x32x32x16xi8>, %arg1: tensor<1x5x5x16xi8>, %arg2: tensor<16xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %arg2) {
    kernel_shape = [5, 5],
    auto_pad = "NOTSET"
  } : (tensor<1x32x32x16xi8>, tensor<1x5x5x16xi8>, tensor<16xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_depthwise_conv_with_bias
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %arg2) {{.*}} -> tensor<1x28x28x16xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x28x28x16xi8>
}

// -----

// COM: Test 2D depthwise conv with SAME_UPPER padding (NHWC layout)
// Input: [N=1, H=28, W=28, C=64], auto_pad: SAME_UPPER
// Output: [N=1, H=28, W=28, C=64] (same spatial size)
func.func @test_XCOMPILER_depthwise_conv_same_pad(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<1x3x3x64xi8>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    auto_pad = "SAME_UPPER"
  } : (tensor<1x28x28x64xi8>, tensor<1x3x3x64xi8>, none) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_depthwise_conv_same_pad
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %0) {{.*}} -> tensor<1x28x28x64xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x28x28x64xi8>
}

// -----

// COM: Test 2D depthwise conv with dilation (NHWC layout)
// Input: [N=1, H=28, W=28, C=32], Kernel: 3x3, Dilation: [2, 2]
// Effective kernel: 5x5, Output: [N=1, H=24, W=24, C=32]
func.func @test_XCOMPILER_depthwise_conv_dilated(%arg0: tensor<1x28x28x32xi8>, %arg1: tensor<1x3x3x32xi8>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    dilations = [2, 2],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x32xi8>, tensor<1x3x3x32xi8>, none) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_depthwise_conv_dilated
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %0) {{.*}} -> tensor<1x24x24x32xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x24x24x32xi8>
}

// -----

// COM: Test 3D depthwise conv (5D tensors, NDHWC layout)
// Input: [N=1, D=16, H=32, W=32, C=32], Weight IDHWO: [M=1, kD=3, kH=3, kW=3, C=32]
// Output: [N=1, D=14, H=30, W=30, C=32]
func.func @test_XCOMPILER_depthwise_conv3d_basic(%arg0: tensor<1x16x32x32x32xi8>, %arg1: tensor<1x3x3x3x32xi8>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3, 3],
    auto_pad = "NOTSET"
  } : (tensor<1x16x32x32x32xi8>, tensor<1x3x3x3x32xi8>, none) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_depthwise_conv3d_basic
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %0) {{.*}} -> tensor<1x14x30x30x32xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x14x30x30x32xi8>
}

// -----

// COM: Test 3D depthwise conv with SAME_UPPER padding (NDHWC layout)
// Input: [N=1, D=8, H=16, W=16, C=64], auto_pad: SAME_UPPER
// Output: [N=1, D=8, H=16, W=16, C=64]
func.func @test_XCOMPILER_depthwise_conv3d_same_pad(%arg0: tensor<1x8x16x16x64xi8>, %arg1: tensor<1x3x3x3x64xi8>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3, 3],
    auto_pad = "SAME_UPPER"
  } : (tensor<1x8x16x16x64xi8>, tensor<1x3x3x3x64xi8>, none) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_depthwise_conv3d_same_pad
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %0) {{.*}} -> tensor<1x8x16x16x64xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x8x16x16x64xi8>
}
