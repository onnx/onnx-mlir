// RUN: onnx-mlir-opt --split-input-file --merge-slice-concat %s | FileCheck %s

// -----

// ============================================================================
// NCHW Layout Tests (Optimizable)
// ============================================================================

// Test Slice->Concat->InstanceNorm->Conv pattern in NCHW layout
// Slices reorder channels: [32:64] then [0:32] -> Concat -> InstanceNorm -> Conv
// Should reorder InstanceNorm params (f32) and Conv weights (f32)

// CHECK-LABEL: func @test_slice_concat_instancenorm_conv_nchw
func.func @test_slice_concat_instancenorm_conv_nchw(%arg0: tensor<1x64x56x56xf32>) -> tensor<1x128x56x56xf32> {
  // Slice parameters for [channels 32:64]
  %starts1 = onnx.Constant dense<[0, 32, 0, 0]> : tensor<4xi64>
  %ends1 = onnx.Constant dense<[1, 64, 56, 56]> : tensor<4xi64>
  %axes1 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps1 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  // Slice parameters for [channels 0:32]
  %starts2 = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
  %ends2 = onnx.Constant dense<[1, 32, 56, 56]> : tensor<4xi64>
  %axes2 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps2 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  // Perform slices - reorder channels
  %slice1 = "onnx.Slice"(%arg0, %starts1, %ends1, %axes1, %steps1) : (tensor<1x64x56x56xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x32x56x56xf32>
  %slice2 = "onnx.Slice"(%arg0, %starts2, %ends2, %axes2, %steps2) : (tensor<1x64x56x56xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x32x56x56xf32>

  // Concat on channel axis (axis 1 for NCHW)
  %concat = "onnx.Concat"(%slice1, %slice2) {axis = 1 : si64} : (tensor<1x32x56x56xf32>, tensor<1x32x56x56xf32>) -> tensor<1x64x56x56xf32>

  // InstanceNorm parameters (f32)
  %scale = onnx.Constant dense<1.5> : tensor<64xf32>
  %bias_in = onnx.Constant dense<0.3> : tensor<64xf32>

  // CHECK-NOT: onnx.Slice
  // CHECK-NOT: onnx.Concat
  // CHECK-DAG: onnx.Constant dense<{{.*}}> : tensor<128x64x3x3xf32>
  // CHECK-DAG: onnx.Constant dense<{{.*}}> : tensor<64xf32>
  // CHECK-DAG: onnx.Constant dense<{{.*}}> : tensor<64xf32>
  // CHECK: "onnx.InstanceNormalization"(%arg0
  %in = "onnx.InstanceNormalization"(%concat, %scale, %bias_in) {
    epsilon = 1.0e-5 : f32
  } : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>

  // Conv with f32 weights in OIHW format [out_ch, in_ch, kH, kW]
  %weights = onnx.Constant dense<2.7> : tensor<128x64x3x3xf32>
  %bias_conv = "onnx.NoValue"() {value} : () -> none

  // CHECK: "onnx.Conv"
  %conv = "onnx.Conv"(%in, %weights, %bias_conv) {
    auto_pad = "NOTSET",
    dilations = [1, 1],
    group = 1 : si64,
    kernel_shape = [3, 3],
    pads = [1, 1, 1, 1],
    strides = [1, 1]
  } : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>, none) -> tensor<1x128x56x56xf32>

  return %conv : tensor<1x128x56x56xf32>
}

// -----

// Test with 3 slices in NCHW layout
// Slices: [0:16], [16:48], [48:64] - in original order (identity)

// CHECK-LABEL: func @test_three_slices_nchw
func.func @test_three_slices_nchw(%arg0: tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32> {
  // Slice 1: [channels 0:16]
  %starts1 = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
  %ends1 = onnx.Constant dense<[1, 16, 28, 28]> : tensor<4xi64>
  %axes1 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps1 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  // Slice 2: [channels 16:48]
  %starts2 = onnx.Constant dense<[0, 16, 0, 0]> : tensor<4xi64>
  %ends2 = onnx.Constant dense<[1, 48, 28, 28]> : tensor<4xi64>
  %axes2 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps2 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  // Slice 3: [channels 48:64]
  %starts3 = onnx.Constant dense<[0, 48, 0, 0]> : tensor<4xi64>
  %ends3 = onnx.Constant dense<[1, 64, 28, 28]> : tensor<4xi64>
  %axes3 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps3 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  %slice1 = "onnx.Slice"(%arg0, %starts1, %ends1, %axes1, %steps1) : (tensor<1x64x28x28xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x16x28x28xf32>
  %slice2 = "onnx.Slice"(%arg0, %starts2, %ends2, %axes2, %steps2) : (tensor<1x64x28x28xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x32x28x28xf32>
  %slice3 = "onnx.Slice"(%arg0, %starts3, %ends3, %axes3, %steps3) : (tensor<1x64x28x28xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x16x28x28xf32>

  // Concat in original order (identity)
  %concat = "onnx.Concat"(%slice1, %slice2, %slice3) {axis = 1 : si64} : (tensor<1x16x28x28xf32>, tensor<1x32x28x28xf32>, tensor<1x16x28x28xf32>) -> tensor<1x64x28x28xf32>

  // InstanceNorm with f32 params
  %scale = onnx.Constant dense<2.1> : tensor<64xf32>
  %bias_in = onnx.Constant dense<0.6> : tensor<64xf32>

  // CHECK-NOT: onnx.Slice
  // CHECK-NOT: onnx.Concat
  // CHECK-DAG: onnx.Constant dense<{{.*}}> : tensor<64x64x1x1xf32>
  // CHECK-DAG: onnx.Constant dense<{{.*}}> : tensor<64xf32>
  // CHECK-DAG: onnx.Constant dense<{{.*}}> : tensor<64xf32>
  // CHECK: "onnx.InstanceNormalization"(%arg0
  %in = "onnx.InstanceNormalization"(%concat, %scale, %bias_in) {
    epsilon = 1.0e-5 : f32
  } : (tensor<1x64x28x28xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x28x28xf32>

  // Conv with f32 OIHW weights
  %weights = onnx.Constant dense<3.2> : tensor<64x64x1x1xf32>
  %bias_conv = "onnx.NoValue"() {value} : () -> none

  // CHECK: "onnx.Conv"
  %conv = "onnx.Conv"(%in, %weights, %bias_conv) {
    auto_pad = "NOTSET",
    dilations = [1, 1],
    group = 1 : si64,
    kernel_shape = [1, 1],
    pads = [0, 0, 0, 0],
    strides = [1, 1]
  } : (tensor<1x64x28x28xf32>, tensor<64x64x1x1xf32>, none) -> tensor<1x64x28x28xf32>

  return %conv : tensor<1x64x28x28xf32>
}

// -----

// Test with 4 slices and complex reordering in NCHW
// Original: [0:16], [16:32], [32:48], [48:64]
// Reordered: [32:48], [0:16], [48:64], [16:32]

// CHECK-LABEL: func @test_four_slices_reordered_nchw
func.func @test_four_slices_reordered_nchw(%arg0: tensor<1x64x32x32xf32>) -> tensor<1x32x32x32xf32> {
  // Slice 1: [channels 0:16]
  %starts1 = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
  %ends1 = onnx.Constant dense<[1, 16, 32, 32]> : tensor<4xi64>
  %axes1 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps1 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  // Slice 2: [channels 16:32]
  %starts2 = onnx.Constant dense<[0, 16, 0, 0]> : tensor<4xi64>
  %ends2 = onnx.Constant dense<[1, 32, 32, 32]> : tensor<4xi64>
  %axes2 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps2 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  // Slice 3: [channels 32:48]
  %starts3 = onnx.Constant dense<[0, 32, 0, 0]> : tensor<4xi64>
  %ends3 = onnx.Constant dense<[1, 48, 32, 32]> : tensor<4xi64>
  %axes3 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps3 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  // Slice 4: [channels 48:64]
  %starts4 = onnx.Constant dense<[0, 48, 0, 0]> : tensor<4xi64>
  %ends4 = onnx.Constant dense<[1, 64, 32, 32]> : tensor<4xi64>
  %axes4 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps4 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  %slice1 = "onnx.Slice"(%arg0, %starts1, %ends1, %axes1, %steps1) : (tensor<1x64x32x32xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x16x32x32xf32>
  %slice2 = "onnx.Slice"(%arg0, %starts2, %ends2, %axes2, %steps2) : (tensor<1x64x32x32xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x16x32x32xf32>
  %slice3 = "onnx.Slice"(%arg0, %starts3, %ends3, %axes3, %steps3) : (tensor<1x64x32x32xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x16x32x32xf32>
  %slice4 = "onnx.Slice"(%arg0, %starts4, %ends4, %axes4, %steps4) : (tensor<1x64x32x32xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x16x32x32xf32>

  // Concat in reordered fashion: [32:48], [0:16], [48:64], [16:32]
  %concat = "onnx.Concat"(%slice3, %slice1, %slice4, %slice2) {axis = 1 : si64} : (tensor<1x16x32x32xf32>, tensor<1x16x32x32xf32>, tensor<1x16x32x32xf32>, tensor<1x16x32x32xf32>) -> tensor<1x64x32x32xf32>

  // InstanceNorm with f32 params
  %scale = onnx.Constant dense<1.8> : tensor<64xf32>
  %bias_in = onnx.Constant dense<0.4> : tensor<64xf32>

  // CHECK-NOT: onnx.Slice
  // CHECK-NOT: onnx.Concat
  // CHECK-DAG: onnx.Constant dense<{{.*}}> : tensor<32x64x3x3xf32>
  // CHECK-DAG: onnx.Constant dense<{{.*}}> : tensor<64xf32>
  // CHECK-DAG: onnx.Constant dense<{{.*}}> : tensor<64xf32>
  // CHECK: "onnx.InstanceNormalization"(%arg0
  %in = "onnx.InstanceNormalization"(%concat, %scale, %bias_in) {
    epsilon = 1.0e-5 : f32
  } : (tensor<1x64x32x32xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x32x32xf32>

  // Conv with f32 OIHW weights
  %weights = onnx.Constant dense<2.9> : tensor<32x64x3x3xf32>
  %bias_conv = "onnx.NoValue"() {value} : () -> none

  // CHECK: "onnx.Conv"
  %conv = "onnx.Conv"(%in, %weights, %bias_conv) {
    auto_pad = "NOTSET",
    dilations = [1, 1],
    group = 1 : si64,
    kernel_shape = [3, 3],
    pads = [1, 1, 1, 1],
    strides = [1, 1]
  } : (tensor<1x64x32x32xf32>, tensor<32x64x3x3xf32>, none) -> tensor<1x32x32x32xf32>

  return %conv : tensor<1x32x32x32xf32>
}

// ============================================================================
// Negative Tests
// ============================================================================

// Negative test: Concat on wrong axis (should not optimize)

// CHECK-LABEL: func @test_wrong_axis
func.func @test_wrong_axis(%arg0: tensor<1x64x56x56xf32>) -> tensor<1x64x112x56xf32> {
  %starts1 = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
  %ends1 = onnx.Constant dense<[1, 64, 28, 56]> : tensor<4xi64>
  %axes1 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps1 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  %starts2 = onnx.Constant dense<[0, 0, 28, 0]> : tensor<4xi64>
  %ends2 = onnx.Constant dense<[1, 64, 56, 56]> : tensor<4xi64>
  %axes2 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps2 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  %slice1 = "onnx.Slice"(%arg0, %starts1, %ends1, %axes1, %steps1) : (tensor<1x64x56x56xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x64x28x56xf32>
  %slice2 = "onnx.Slice"(%arg0, %starts2, %ends2, %axes2, %steps2) : (tensor<1x64x56x56xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x64x28x56xf32>

  // Concat on axis 2 (height) instead of axis 1 (channels) - should NOT optimize
  // CHECK: onnx.Slice
  // CHECK: onnx.Slice
  // CHECK: onnx.Concat
  %concat = "onnx.Concat"(%slice1, %slice2) {axis = 2 : si64} : (tensor<1x64x28x56xf32>, tensor<1x64x28x56xf32>) -> tensor<1x64x112x56xf32>

  %scale = onnx.Constant dense<1.1> : tensor<64xf32>
  %bias_in = onnx.Constant dense<0.1> : tensor<64xf32>

  %in = "onnx.InstanceNormalization"(%concat, %scale, %bias_in) {
    epsilon = 1.0e-5 : f32
  } : (tensor<1x64x112x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x56xf32>

  %weights = onnx.Constant dense<2.2> : tensor<64x64x3x3xf32>
  %bias_conv = "onnx.NoValue"() {value} : () -> none

  %conv = "onnx.Conv"(%in, %weights, %bias_conv) {
    auto_pad = "NOTSET",
    dilations = [1, 1],
    group = 1 : si64,
    kernel_shape = [3, 3],
    pads = [1, 1, 1, 1],
    strides = [1, 1]
  } : (tensor<1x64x112x56xf32>, tensor<64x64x3x3xf32>, none) -> tensor<1x64x112x56xf32>

  return %conv : tensor<1x64x112x56xf32>
}

// -----

// Negative test: Missing InstanceNorm (direct Slice->Concat->Conv)
// Should NOT optimize as pattern requires InstanceNorm

// CHECK-LABEL: func @test_missing_instancenorm
func.func @test_missing_instancenorm(%arg0: tensor<1x64x56x56xf32>) -> tensor<1x128x56x56xf32> {
  %starts1 = onnx.Constant dense<[0, 32, 0, 0]> : tensor<4xi64>
  %ends1 = onnx.Constant dense<[1, 64, 56, 56]> : tensor<4xi64>
  %axes1 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps1 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  %starts2 = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
  %ends2 = onnx.Constant dense<[1, 32, 56, 56]> : tensor<4xi64>
  %axes2 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  %steps2 = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>

  %slice1 = "onnx.Slice"(%arg0, %starts1, %ends1, %axes1, %steps1) : (tensor<1x64x56x56xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x32x56x56xf32>
  %slice2 = "onnx.Slice"(%arg0, %starts2, %ends2, %axes2, %steps2) : (tensor<1x64x56x56xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x32x56x56xf32>

  %concat = "onnx.Concat"(%slice1, %slice2) {axis = 1 : si64} : (tensor<1x32x56x56xf32>, tensor<1x32x56x56xf32>) -> tensor<1x64x56x56xf32>

  // No InstanceNorm - pattern should NOT match
  // CHECK: onnx.Slice
  // CHECK: onnx.Slice
  // CHECK: onnx.Concat
  // CHECK: onnx.Conv
  %weights = onnx.Constant dense<3.5> : tensor<128x64x3x3xf32>
  %bias = "onnx.NoValue"() {value} : () -> none

  %conv = "onnx.Conv"(%concat, %weights, %bias) {
    auto_pad = "NOTSET",
    dilations = [1, 1],
    group = 1 : si64,
    kernel_shape = [3, 3],
    pads = [1, 1, 1, 1],
    strides = [1, 1]
  } : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>, none) -> tensor<1x128x56x56xf32>

  return %conv : tensor<1x128x56x56xf32>
}

// -----

// ============================================================================
// Quantized Type Tests (Native Quant Types)
// ============================================================================

// Test Slice->Concat->InstanceNorm->Conv with native quantized types
// This validates the pass handles quant.uniform types directly

// CHECK-LABEL: func.func @test_slice_concat_instancenorm_conv_quantized
func.func @test_slice_concat_instancenorm_conv_quantized(%arg0: tensor<1x16x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>) -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.023529411764705882:64>> {
  // Slice parameters for [channels 8:16]
  %starts1 = "onnx.Constant"() {value = dense<[0, 8, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %ends1 = "onnx.Constant"() {value = dense<[1, 16, 8, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
  %axes1 = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  %steps1 = "onnx.Constant"() {value = dense<[1, 1, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>

  // Slice parameters for [channels 0:8]
  %starts2 = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %ends2 = "onnx.Constant"() {value = dense<[1, 8, 8, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
  %axes2 = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  %steps2 = "onnx.Constant"() {value = dense<[1, 1, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>

  // Perform slices - reorder channels from [8:16][0:8]
  %slice1 = "onnx.Slice"(%arg0, %starts1, %ends1, %axes1, %steps1) : (tensor<1x16x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>
  %slice2 = "onnx.Slice"(%arg0, %starts2, %ends2, %axes2, %steps2) : (tensor<1x16x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>

  // Concat on channel axis (axis 1 for NCHW)
  %concat = "onnx.Concat"(%slice1, %slice2) {axis = 1 : si64} : (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>, tensor<1x8x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>) -> tensor<1x16x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>

  // InstanceNorm parameters as f32
  %scale = "onnx.Constant"() {value = dense<1.5> : tensor<16xf32>} : () -> tensor<16xf32>
  %bias_in = "onnx.Constant"() {value = dense<0.3> : tensor<16xf32>} : () -> tensor<16xf32>

  // CHECK-NOT: onnx.Slice
  // CHECK-NOT: onnx.Concat
  // CHECK-DAG: %[[WEIGHTS:.+]] = onnx.Constant {value = dense<{{.*}}> : tensor<8x16x3x3xf32>} : tensor<8x16x3x3x!quant.uniform<u8:f32, {{.*}}>>
  // CHECK-DAG: %[[BIAS:.+]] = onnx.Constant {value = dense<{{.*}}> : tensor<8xf32>} : tensor<8x!quant.uniform<i32:f32, {{.*}}>>
  // CHECK-DAG: %[[SCALE:.+]] = onnx.Constant dense<{{.*}}> : tensor<16xf32>
  // CHECK-DAG: %[[BIAS_IN:.+]] = onnx.Constant dense<{{.*}}> : tensor<16xf32>
  // CHECK: %[[IN:.+]] = "onnx.InstanceNormalization"(%arg0, %[[SCALE]], %[[BIAS_IN]])
  %in = "onnx.InstanceNormalization"(%concat, %scale, %bias_in) {
    epsilon = 1.0e-5 : f32
  } : (tensor<1x16x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x16x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>

  // Conv weights with quantization - raw data as f32, result type quantized
  %weights = "onnx.Constant"() {value = dense<2.0> : tensor<8x16x3x3xf32>} : () -> tensor<8x16x3x3x!quant.uniform<u8:f32, 0.015686274509803921:0>>

  // Bias with quantization - scale = input_scale * weight_scale
  %bias = "onnx.Constant"() {value = dense<0.5> : tensor<8xf32>} : () -> tensor<8x!quant.uniform<i32:f32, 1.2302086999999999E-4:0>>

  // CHECK: %[[CONV:.+]] = "onnx.Conv"(%[[IN]], %[[WEIGHTS]], %[[BIAS]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x16x8x8x!quant.uniform<u8:f32, {{.*}}:128>>, tensor<8x16x3x3x!quant.uniform<u8:f32, {{.*}}>>, tensor<8x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x8x8x8x!quant.uniform<u8:f32, {{.*}}:64>>
  %conv = "onnx.Conv"(%in, %weights, %bias) {
    auto_pad = "NOTSET",
    dilations = [1, 1],
    group = 1 : si64,
    kernel_shape = [3, 3],
    pads = [1, 1, 1, 1],
    strides = [1, 1]
  } : (tensor<1x16x8x8x!quant.uniform<u8:f32, 0.0078431372549019607:128>>, tensor<8x16x3x3x!quant.uniform<u8:f32, 0.015686274509803921:0>>, tensor<8x!quant.uniform<i32:f32, 1.2302086999999999E-4:0>>) -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.023529411764705882:64>>

  // CHECK: return %[[CONV]]
  return %conv : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.023529411764705882:64>>
}
