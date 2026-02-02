// RUN: onnx-mlir-opt --merge-strided-slice-concat-conv %s -split-input-file | FileCheck %s

// -----
// Test: Pure NCHW pattern - StridedSlice->Concat->Conv (NO transpose)
// Input is NCHW [N,C,H,W], slices on H and W dimensions, concat on C, Conv directly on NCHW

// CHECK-LABEL: func.func @test_pure_nchw_pattern
func.func @test_pure_nchw_pattern(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x8x4x4xf32> {
  // Define slice parameters for NCHW [N, C, H, W]
  %starts_l = "onnx.Constant"() {value = dense<[0, 0, 1, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_r = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_w1 = "onnx.Constant"() {value = dense<[0, 0, 0, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_w0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>

  %ends = "onnx.Constant"() {value = dense<[1, 4, 8, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
  %axes = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>

  %strides_h = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  %strides_w = "onnx.Constant"() {value = dense<[1, 1, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>

  // First level slices (stride on H dimension, axis 2)
  %slice_l = "onnx.Slice"(%arg0, %starts_l, %ends, %axes, %strides_h) : (tensor<1x4x8x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x8xf32>
  %slice_r = "onnx.Slice"(%arg0, %starts_r, %ends, %axes, %strides_h) : (tensor<1x4x8x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x8xf32>

  // Second level slices (stride on W dimension, axis 3)
  %slice_ll = "onnx.Slice"(%slice_l, %starts_w1, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_lr = "onnx.Slice"(%slice_l, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_rl = "onnx.Slice"(%slice_r, %starts_w1, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_rr = "onnx.Slice"(%slice_r, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>

  // Concat in order: rr, lr, rl, ll (axis=1, channel dimension in NCHW)
  %concat = "onnx.Concat"(%slice_rr, %slice_lr, %slice_rl, %slice_ll) {axis = 1 : si64} : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x16x4x4xf32>

  // Conv weights: OIHW format [8, 16, 3, 3]
  %weights = "onnx.Constant"() {value = dense<2.0> : tensor<8x16x3x3xf32>} : () -> tensor<8x16x3x3xf32>
  %bias = "onnx.Constant"() {value = dense<0.5> : tensor<8xf32>} : () -> tensor<8xf32>

  // Conv with kernel=3x3, stride=1, dilation=1 (direct NCHW input, no transpose)
  %conv = "onnx.Conv"(%concat, %weights, %bias) {
    dilations = [1, 1],
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [1, 1, 1, 1]
  } : (tensor<1x16x4x4xf32>, tensor<8x16x3x3xf32>, tensor<8xf32>) -> tensor<1x8x4x4xf32>

  // Pattern should be optimized - slices and concat eliminated
  // CHECK-NOT: onnx.Slice
  // CHECK-NOT: onnx.Concat
  // CHECK: %[[BIAS:.+]] = onnx.Constant dense<5.000000e-01> : tensor<8xf32>
  // CHECK: %[[WEIGHTS:.+]] = onnx.Constant dense<2.000000e+00> : tensor<8x4x6x6xf32>
  // CHECK: %[[CONV:.+]] = "onnx.Conv"(%arg0, %[[WEIGHTS]], %[[BIAS]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x4x8x8xf32>, tensor<8x4x6x6xf32>, tensor<8xf32>) -> tensor<1x8x4x4xf32>
  // CHECK: return %[[CONV]]

  return %conv : tensor<1x8x4x4xf32>
}

// -----
// Test 2: Negative - Wrong concat order (should NOT optimize)

// CHECK-LABEL: func.func @test_wrong_concat_order
func.func @test_wrong_concat_order(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x8x4x4xf32> {
  %starts_l = "onnx.Constant"() {value = dense<[0, 0, 1, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_r = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_w1 = "onnx.Constant"() {value = dense<[0, 0, 0, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_w0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>

  %ends = "onnx.Constant"() {value = dense<[1, 4, 8, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
  %axes = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>

  %strides_h = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  %strides_w = "onnx.Constant"() {value = dense<[1, 1, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>

  %slice_l = "onnx.Slice"(%arg0, %starts_l, %ends, %axes, %strides_h) : (tensor<1x4x8x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x8xf32>
  %slice_r = "onnx.Slice"(%arg0, %starts_r, %ends, %axes, %strides_h) : (tensor<1x4x8x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x8xf32>

  %slice_ll = "onnx.Slice"(%slice_l, %starts_w1, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_lr = "onnx.Slice"(%slice_l, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_rl = "onnx.Slice"(%slice_r, %starts_w1, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_rr = "onnx.Slice"(%slice_r, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>

  // Wrong order: ll, lr, rl, rr instead of rr, lr, rl, ll
  %concat = "onnx.Concat"(%slice_ll, %slice_lr, %slice_rl, %slice_rr) {axis = 1 : si64} : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x16x4x4xf32>

  %weights = "onnx.Constant"() {value = dense<2.0> : tensor<8x16x3x3xf32>} : () -> tensor<8x16x3x3xf32>
  %bias = "onnx.Constant"() {value = dense<0.5> : tensor<8xf32>} : () -> tensor<8xf32>

  %conv = "onnx.Conv"(%concat, %weights, %bias) {
    dilations = [1, 1],
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [1, 1, 1, 1]
  } : (tensor<1x16x4x4xf32>, tensor<8x16x3x3xf32>, tensor<8xf32>) -> tensor<1x8x4x4xf32>

  // Should NOT optimize due to wrong concat order
  // CHECK: onnx.Slice
  // CHECK: onnx.Concat

  return %conv : tensor<1x8x4x4xf32>
}

// -----
// Test 3: Negative - Wrong stride value (should NOT optimize)

// CHECK-LABEL: func.func @test_wrong_stride
func.func @test_wrong_stride(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x8x3x4xf32> {
  %starts_l = "onnx.Constant"() {value = dense<[0, 0, 1, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_r = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_w1 = "onnx.Constant"() {value = dense<[0, 0, 0, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_w0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>

  %ends = "onnx.Constant"() {value = dense<[1, 4, 8, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
  %axes = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>

  // Wrong stride: [1, 1, 3, 1] instead of [1, 1, 2, 1]
  %strides_h = "onnx.Constant"() {value = dense<[1, 1, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  %strides_w = "onnx.Constant"() {value = dense<[1, 1, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>

  %slice_l = "onnx.Slice"(%arg0, %starts_l, %ends, %axes, %strides_h) : (tensor<1x4x8x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x3x8xf32>
  %slice_r = "onnx.Slice"(%arg0, %starts_r, %ends, %axes, %strides_h) : (tensor<1x4x8x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x3x8xf32>

  %slice_ll = "onnx.Slice"(%slice_l, %starts_w1, %ends, %axes, %strides_w) : (tensor<1x4x3x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x3x4xf32>
  %slice_lr = "onnx.Slice"(%slice_l, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x3x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x3x4xf32>
  %slice_rl = "onnx.Slice"(%slice_r, %starts_w1, %ends, %axes, %strides_w) : (tensor<1x4x3x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x3x4xf32>
  %slice_rr = "onnx.Slice"(%slice_r, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x3x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x3x4xf32>

  %concat = "onnx.Concat"(%slice_rr, %slice_lr, %slice_rl, %slice_ll) {axis = 1 : si64} : (tensor<1x4x3x4xf32>, tensor<1x4x3x4xf32>, tensor<1x4x3x4xf32>, tensor<1x4x3x4xf32>) -> tensor<1x16x3x4xf32>

  %weights = "onnx.Constant"() {value = dense<2.0> : tensor<8x16x3x3xf32>} : () -> tensor<8x16x3x3xf32>
  %bias = "onnx.Constant"() {value = dense<0.5> : tensor<8xf32>} : () -> tensor<8xf32>

  %conv = "onnx.Conv"(%concat, %weights, %bias) {
    dilations = [1, 1],
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [1, 1, 1, 1]
  } : (tensor<1x16x3x4xf32>, tensor<8x16x3x3xf32>, tensor<8xf32>) -> tensor<1x8x3x4xf32>

  // Should NOT optimize due to wrong stride
  // CHECK: onnx.Slice
  // CHECK: onnx.Concat

  return %conv : tensor<1x8x3x4xf32>
}

// -----
// Test 4: Negative - Only 3 slices instead of 4 (should NOT optimize)

// CHECK-LABEL: func.func @test_only_three_slices
func.func @test_only_three_slices(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x8x4x4xf32> {
  %starts_l = "onnx.Constant"() {value = dense<[0, 0, 1, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_r = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_w0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>

  %ends = "onnx.Constant"() {value = dense<[1, 4, 8, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
  %axes = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>

  %strides_h = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  %strides_w = "onnx.Constant"() {value = dense<[1, 1, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>

  %slice_l = "onnx.Slice"(%arg0, %starts_l, %ends, %axes, %strides_h) : (tensor<1x4x8x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x8xf32>
  %slice_r = "onnx.Slice"(%arg0, %starts_r, %ends, %axes, %strides_h) : (tensor<1x4x8x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x8xf32>

  // Only 3 second-level slices instead of 4
  %slice_lr = "onnx.Slice"(%slice_l, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_rl = "onnx.Slice"(%slice_r, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_rr = "onnx.Slice"(%slice_r, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>

  // Only 3 inputs to concat instead of 4
  %concat = "onnx.Concat"(%slice_rr, %slice_lr, %slice_rl) {axis = 1 : si64} : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x12x4x4xf32>

  %weights = "onnx.Constant"() {value = dense<2.0> : tensor<8x12x3x3xf32>} : () -> tensor<8x12x3x3xf32>
  %bias = "onnx.Constant"() {value = dense<0.5> : tensor<8xf32>} : () -> tensor<8xf32>

  %conv = "onnx.Conv"(%concat, %weights, %bias) {
    dilations = [1, 1],
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [1, 1, 1, 1]
  } : (tensor<1x12x4x4xf32>, tensor<8x12x3x3xf32>, tensor<8xf32>) -> tensor<1x8x4x4xf32>

  // Should NOT optimize due to only 3 concat inputs instead of 4
  // CHECK: onnx.Slice
  // CHECK: onnx.Concat

  return %conv : tensor<1x8x4x4xf32>
}

// -----
// Test 5: Negative - Conv stride != 1 (should NOT optimize)

// CHECK-LABEL: func.func @test_conv_stride_not_one
func.func @test_conv_stride_not_one(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x8x2x2xf32> {
  %starts_l = "onnx.Constant"() {value = dense<[0, 0, 1, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_r = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_w1 = "onnx.Constant"() {value = dense<[0, 0, 0, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts_w0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>

  %ends = "onnx.Constant"() {value = dense<[1, 4, 8, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
  %axes = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>

  %strides_h = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  %strides_w = "onnx.Constant"() {value = dense<[1, 1, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>

  %slice_l = "onnx.Slice"(%arg0, %starts_l, %ends, %axes, %strides_h) : (tensor<1x4x8x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x8xf32>
  %slice_r = "onnx.Slice"(%arg0, %starts_r, %ends, %axes, %strides_h) : (tensor<1x4x8x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x8xf32>

  %slice_ll = "onnx.Slice"(%slice_l, %starts_w1, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_lr = "onnx.Slice"(%slice_l, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_rl = "onnx.Slice"(%slice_r, %starts_w1, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  %slice_rr = "onnx.Slice"(%slice_r, %starts_w0, %ends, %axes, %strides_w) : (tensor<1x4x4x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x4x4xf32>

  %concat = "onnx.Concat"(%slice_rr, %slice_lr, %slice_rl, %slice_ll) {axis = 1 : si64} : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x16x4x4xf32>

  %weights = "onnx.Constant"() {value = dense<2.0> : tensor<8x16x3x3xf32>} : () -> tensor<8x16x3x3xf32>
  %bias = "onnx.Constant"() {value = dense<0.5> : tensor<8xf32>} : () -> tensor<8xf32>

  %conv = "onnx.Conv"(%concat, %weights, %bias) {
    dilations = [1, 1],
    kernel_shape = [3, 3],
    strides = [2, 2],
    pads = [1, 1, 1, 1]
  } : (tensor<1x16x4x4xf32>, tensor<8x16x3x3xf32>, tensor<8xf32>) -> tensor<1x8x2x2xf32>

  // Should NOT optimize due to conv stride != 1
  // CHECK: onnx.Slice
  // CHECK: onnx.Concat

  return %conv : tensor<1x8x2x2xf32>
}
