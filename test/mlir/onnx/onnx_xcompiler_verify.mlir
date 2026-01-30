// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
/// Verification tests for XCOMPILER Operations
/// Domain: com.amd.xcompiler
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// XCOMPILER FusedEltwise Tests (Quantized Element-wise Operations)
//===----------------------------------------------------------------------===//

// Test: clip_min should fail when type is not CLIP
func.func @test_clip_attr_invalid(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8> {
  // expected-error @+1 {{'clip_min' is only valid when type is CLIP}}
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {type = "ADD", nonlinear = "NONE", clip_min = 0 : si64} : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  onnx.Return %0 : tensor<4xui8>
}

// ----- 

// Test: leakyrelu_alpha should fail when nonlinear is not LEAKYRELU  
func.func @test_leakyrelu_attr_invalid(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8> {
  // expected-error @+1 {{'leakyrelu_alpha' is only valid when nonlinear is LEAKYRELU}}
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {type = "ADD", nonlinear = "RELU", leakyrelu_alpha = 0.1 : f32} : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  onnx.Return %0 : tensor<4xui8>
}

// ----- 

// Test: nonlinear_in_scales should fail when nonlinear is NONE
func.func @test_nonlinear_scales_invalid(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8>  {
  // expected-error @+1 {{'nonlinear_in_scales' is only valid when nonlinear is not NONE}}
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {type = "ADD", nonlinear = "NONE", nonlinear_in_scales = 1.0 : f32} : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  onnx.Return %0 : tensor<4xui8>
}

// ----- 

// Test: Valid op (should pass)
func.func @test_valid_clip(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {type = "CLIP", nonlinear = "NONE", clip_min = 0 : si64, clip_max = 255 : si64} : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  onnx.Return %0 : tensor<4xui8>
}

//===----------------------------------------------------------------------===//
/// XCOMPILER DepthwiseConv Tests (Depthwise Separable Convolution - NHWC layout)
/// Supports both 2D (4D tensors) and 3D (5D tensors)
/// Weight format: OHWI [C, kH, kW, 1] for 2D, [C, kD, kH, kW, 1] for 3D
//===----------------------------------------------------------------------===//

// -----

// Test: 2D depthwise conv - kernel_shape must have 2 elements for 4D input
// Input: [N, H, W, C], Weight OHWI: [C, kH, kW, M]
func.func @test_depthwise_conv_invalid_kernel_shape(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<64x3x3x1xi8>) -> tensor<1x26x26x64xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{kernel_shape must have 2 elements for 2D convolution, got 3}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3, 3],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<64x3x3x1xi8>, none) -> tensor<1x26x26x64xi8>
  onnx.Return %0 : tensor<1x26x26x64xi8>
}

// -----

// Test: 2D depthwise conv - strides must have 2 elements for 4D input
func.func @test_depthwise_conv_invalid_strides(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<64x3x3x1xi8>) -> tensor<1x26x26x64xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{strides must have 2 elements, got 1}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    strides = [1],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<64x3x3x1xi8>, none) -> tensor<1x26x26x64xi8>
  onnx.Return %0 : tensor<1x26x26x64xi8>
}

// -----

// Test: 2D depthwise conv - dilations must have 2 elements for 4D input
func.func @test_depthwise_conv_invalid_dilations(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<64x3x3x1xi8>) -> tensor<1x26x26x64xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{dilations must have 2 elements, got 3}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    dilations = [1, 1, 1],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<64x3x3x1xi8>, none) -> tensor<1x26x26x64xi8>
  onnx.Return %0 : tensor<1x26x26x64xi8>
}

// -----

// Test: 2D depthwise conv - pads must have 4 elements (2 * num_spatial_dims)
func.func @test_depthwise_conv_invalid_pads(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<64x3x3x1xi8>) -> tensor<1x28x28x64xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{pads must have 4 elements, got 2}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    pads = [1, 1],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<64x3x3x1xi8>, none) -> tensor<1x28x28x64xi8>
  onnx.Return %0 : tensor<1x28x28x64xi8>
}

// -----

// Test: auto_pad must be valid value
func.func @test_depthwise_conv_invalid_auto_pad(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<64x3x3x1xi8>) -> tensor<1x26x26x64xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{auto_pad must be one of NOTSET, SAME_UPPER, SAME_LOWER, VALID, got 'INVALID'}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    auto_pad = "INVALID"
  } : (tensor<1x28x28x64xi8>, tensor<64x3x3x1xi8>, none) -> tensor<1x26x26x64xi8>
  onnx.Return %0 : tensor<1x26x26x64xi8>
}

// -----

// Test: input must be 4D or 5D (NHWC or NDHWC layout)
func.func @test_depthwise_conv_invalid_input_rank(%arg0: tensor<28x28x64xi8>, %arg1: tensor<64x3x3x1xi8>) -> tensor<26x26x64xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{input X must be 4D [N, H, W, C] or 5D [N, D, H, W, C], got rank 3}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    auto_pad = "NOTSET"
  } : (tensor<28x28x64xi8>, tensor<64x3x3x1xi8>, none) -> tensor<26x26x64xi8>
  onnx.Return %0 : tensor<26x26x64xi8>
}

// -----

// Test: 2D weight must be 4D [C, kH, kW, M] in OHWI format
func.func @test_depthwise_conv_invalid_weight_rank(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<64x3x3xi8>) -> tensor<1x26x26x64xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{weight W must be 4D tensor for 2D convolution, got rank 3}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<64x3x3xi8>, none) -> tensor<1x26x26x64xi8>
  onnx.Return %0 : tensor<1x26x26x64xi8>
}

// -----

// Test: bias must be 1D
func.func @test_depthwise_conv_invalid_bias_rank(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<64x3x3x1xi8>, %arg2: tensor<1x64xi8>) -> tensor<1x26x26x64xi8> {
  // expected-error @+1 {{bias B must be 1D tensor [C], got rank 2}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %arg2) {
    kernel_shape = [3, 3],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<64x3x3x1xi8>, tensor<1x64xi8>) -> tensor<1x26x26x64xi8>
  onnx.Return %0 : tensor<1x26x26x64xi8>
}

// -----

// Test: depthwise conv weight channel multiplier must be 1 (last dim in OHWI)
// Weight OHWI: [C, kH, kW, M] - M should be 1
func.func @test_depthwise_conv_invalid_channel_multiplier(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<64x3x3x2xi8>) -> tensor<1x26x26x64xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{depthwise conv weight channel multiplier (last dim) should be 1, got 2}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<64x3x3x2xi8>, none) -> tensor<1x26x26x64xi8>
  onnx.Return %0 : tensor<1x26x26x64xi8>
}

// -----

// Test: input channels must match weight channels (first dim in OHWI)
// Input: [N, H, W, C=64], Weight OHWI: [C=32, kH, kW, M]
func.func @test_depthwise_conv_channel_mismatch(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<32x3x3x1xi8>) -> tensor<1x26x26x64xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{input channels (64) must match weight channels (32)}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<32x3x3x1xi8>, none) -> tensor<1x26x26x64xi8>
  onnx.Return %0 : tensor<1x26x26x64xi8>
}

// -----

// Test: Valid 2D depthwise conv (NHWC input, OHWI weight - should pass)
func.func @test_valid_depthwise_conv(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<64x3x3x1xi8>) -> tensor<1x26x26x64xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [0, 0, 0, 0],
    dilations = [1, 1],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<64x3x3x1xi8>, none) -> tensor<1x26x26x64xi8>
  onnx.Return %0 : tensor<1x26x26x64xi8>
}

// -----

// Test: Valid 2D depthwise conv with bias (NHWC input, OHWI weight - should pass)
func.func @test_valid_depthwise_conv_with_bias(%arg0: tensor<1x28x28x64xi8>, %arg1: tensor<64x3x3x1xi8>, %arg2: tensor<64xi8>) -> tensor<1x26x26x64xi8> {
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %arg2) {
    kernel_shape = [3, 3],
    auto_pad = "NOTSET"
  } : (tensor<1x28x28x64xi8>, tensor<64x3x3x1xi8>, tensor<64xi8>) -> tensor<1x26x26x64xi8>
  onnx.Return %0 : tensor<1x26x26x64xi8>
}

// -----

// Test: Valid 3D depthwise conv (NDHWC input, OHWI weight - should pass)
func.func @test_valid_depthwise_conv3d(%arg0: tensor<1x16x28x28x32xi8>, %arg1: tensor<32x3x3x3x1xi8>) -> tensor<1x14x26x26x32xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3, 3],
    strides = [1, 1, 1],
    pads = [0, 0, 0, 0, 0, 0],
    dilations = [1, 1, 1],
    auto_pad = "NOTSET"
  } : (tensor<1x16x28x28x32xi8>, tensor<32x3x3x3x1xi8>, none) -> tensor<1x14x26x26x32xi8>
  onnx.Return %0 : tensor<1x14x26x26x32xi8>
}

// -----

// Test: 3D depthwise conv - kernel_shape must have 3 elements for 5D input
func.func @test_depthwise_conv3d_invalid_kernel_shape(%arg0: tensor<1x16x28x28x32xi8>, %arg1: tensor<32x3x3x3x1xi8>) -> tensor<1x14x26x26x32xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{kernel_shape must have 3 elements for 3D convolution, got 2}}
  %0 = "onnx.XCOMPILERDepthwiseConv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    auto_pad = "NOTSET"
  } : (tensor<1x16x28x28x32xi8>, tensor<32x3x3x3x1xi8>, none) -> tensor<1x14x26x26x32xi8>
  onnx.Return %0 : tensor<1x14x26x26x32xi8>
}

// -----

// Test: 3D depthwise conv - pads must have 6 elements for 5D input
func.func @test_depthwise_conv3d_invalid_pads(%arg0: tensor<1x16x28x28x32xi8>, %arg1: tensor<32x3x3x3x1xi8>) -> tensor<1x16x28x28x32xi8> {
  % none = "onnx.NoValue"(){value} : ()->none
           // expected-error @+1 {{pads must have 6 elements, got 4}}
           % 0 = "onnx.XCOMPILERDepthwiseConv"(
                     % arg0, % arg1, % none){kernel_shape = [ 3, 3, 3 ],
                     pads = [ 1, 1, 1, 1 ], auto_pad = "NOTSET"}
      : (tensor<1x16x28x28x32xi8>, tensor<32x3x3x3x1xi8>, none)
            ->tensor<1x16x28x28x32xi8>
                onnx.Return %
                 0 : tensor<1x16x28x28x32xi8>
}
