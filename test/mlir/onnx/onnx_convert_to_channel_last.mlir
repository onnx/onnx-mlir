// RUN: onnx-mlir-opt --convert-to-channel-last --shape-inference %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test ONNX to ChannelLast Conversions
/// - Conv -> XFEConv
/// - AveragePool -> XFEAveragePool
/// - MaxPool -> XFEMaxPool
/// - GlobalAveragePool -> XFEGlobalAveragePool
/// - GlobalMaxPool -> XFEGlobalMaxPool
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// Conv Conversion Tests
//===----------------------------------------------------------------------===//

// COM: Test basic 2D convolution conversion
// CHECK-LABEL: func.func @test_conv_to_conv_channel_last
func.func @test_conv_to_conv_channel_last(%arg0: tensor<1x3x28x28xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x26x26xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
    dilations = [1, 1],
    group = 1 : si64,
    pads = [0, 0, 0, 0],
    strides = [1, 1]
  } : (tensor<1x3x28x28xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x26x26xf32>
  onnx.Return %0 : tensor<1x64x26x26xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x28x28xf32>) -> tensor<1x28x28x3xf32>
  // CHECK: [[WEIGHT_HWIO:%.+]] = "onnx.Transpose"(%arg1) {perm = [2, 3, 1, 0]} : (tensor<64x3x3x3xf32>) -> tensor<3x3x3x64xf32>
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_HWIO]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<3x3x3x64xf32>, tensor<64xf32>) -> tensor<1x26x26x64xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[CONV_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x26x26x64xf32>) -> tensor<1x64x26x26xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x26x26xf32>
}

// -----

// COM: Test convolution with padding
// CHECK-LABEL: func.func @test_conv_to_conv_channel_last_padded
func.func @test_conv_to_conv_channel_last_padded(%arg0: tensor<1x3x28x28xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x28x28xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
    dilations = [1, 1],
    group = 1 : si64,
    pads = [1, 1, 1, 1],
    strides = [1, 1]
  } : (tensor<1x3x28x28xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x28x28xf32>
  onnx.Return %0 : tensor<1x64x28x28xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x28x28xf32>) -> tensor<1x28x28x3xf32>
  // CHECK: [[WEIGHT_HWIO:%.+]] = "onnx.Transpose"(%arg1) {perm = [2, 3, 1, 0]} : (tensor<64x3x3x3xf32>) -> tensor<3x3x3x64xf32>
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_HWIO]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<3x3x3x64xf32>, tensor<64xf32>) -> tensor<1x28x28x64xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[CONV_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x28x28x64xf32>) -> tensor<1x64x28x28xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x28x28xf32>
}

// -----

// COM: Test convolution with stride
// CHECK-LABEL: func.func @test_conv_to_conv_channel_last_strided
func.func @test_conv_to_conv_channel_last_strided(%arg0: tensor<1x3x28x28xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x13x13xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
    dilations = [1, 1],
    group = 1 : si64,
    pads = [0, 0, 0, 0],
    strides = [2, 2]
  } : (tensor<1x3x28x28xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x13x13xf32>
  onnx.Return %0 : tensor<1x64x13x13xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x28x28xf32>) -> tensor<1x28x28x3xf32>
  // CHECK: [[WEIGHT_HWIO:%.+]] = "onnx.Transpose"(%arg1) {perm = [2, 3, 1, 0]} : (tensor<64x3x3x3xf32>) -> tensor<3x3x3x64xf32>
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_HWIO]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x28x28x3xf32>, tensor<3x3x3x64xf32>, tensor<64xf32>) -> tensor<1x13x13x64xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[CONV_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x13x13x64xf32>) -> tensor<1x64x13x13xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x13x13xf32>
}

// -----

// COM: Test 3D convolution conversion (5D tensors)
// CHECK-LABEL: func.func @test_conv3d_to_conv_channel_last
func.func @test_conv3d_to_conv_channel_last(%arg0: tensor<1x3x8x28x28xf32>, %arg1: tensor<64x3x3x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x6x26x26xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
    dilations = [1, 1, 1],
    group = 1 : si64,
    pads = [0, 0, 0, 0, 0, 0],
    strides = [1, 1, 1]
  } : (tensor<1x3x8x28x28xf32>, tensor<64x3x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x6x26x26xf32>
  onnx.Return %0 : tensor<1x64x6x26x26xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 4, 1]} : (tensor<1x3x8x28x28xf32>) -> tensor<1x8x28x28x3xf32>
  // CHECK: [[WEIGHT_DHWIO:%.+]] = "onnx.Transpose"(%arg1) {perm = [2, 3, 4, 1, 0]} : (tensor<64x3x3x3x3xf32>) -> tensor<3x3x3x3x64xf32>
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_DHWIO]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, pads = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} : (tensor<1x8x28x28x3xf32>, tensor<3x3x3x3x64xf32>, tensor<64xf32>) -> tensor<1x6x26x26x64xf32>
  // CHECK: [[OUTPUT_NCDHW:%.+]] = "onnx.Transpose"([[CONV_CHANNEL_LAST]]) {perm = [0, 4, 1, 2, 3]} : (tensor<1x6x26x26x64xf32>) -> tensor<1x64x6x26x26xf32>
  // CHECK: onnx.Return [[OUTPUT_NCDHW]] : tensor<1x64x6x26x26xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Pooling Conversion Tests
//===----------------------------------------------------------------------===//

// COM: Test AveragePool to AveragePoolChannelLast conversion
// CHECK-LABEL: func.func @test_avgpool_to_channel_last
func.func @test_avgpool_to_channel_last(%arg0: tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32> {
  %0 = "onnx.AveragePool"(%arg0) {
    kernel_shape = [2, 2],
    strides = [2, 2],
    pads = [0, 0, 0, 0]
  } : (tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32>
  onnx.Return %0 : tensor<1x64x14x14xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x28x28xf32>) -> tensor<1x28x28x64xf32>
  // CHECK: [[POOL_CHANNEL_LAST:%.+]] = "onnx.XFEAveragePool"([[INPUT_CHANNEL_LAST]]) {auto_pad = "NOTSET", ceil_mode = 0 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x28x28x64xf32>) -> tensor<1x14x14x64xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[POOL_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x14x14x64xf32>) -> tensor<1x64x14x14xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x14x14xf32>
}

// -----

// COM: Test MaxPool to MaxPoolChannelLast conversion
// CHECK-LABEL: func.func @test_maxpool_to_channel_last
func.func @test_maxpool_to_channel_last(%arg0: tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {
    kernel_shape = [2, 2],
    strides = [2, 2],
    pads = [0, 0, 0, 0],
    dilations = [1, 1]
  } : (tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32>
  onnx.Return %0 : tensor<1x64x14x14xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x28x28xf32>) -> tensor<1x28x28x64xf32>
  // CHECK: [[POOL_CHANNEL_LAST:%.+]] = "onnx.XFEMaxPool"([[INPUT_CHANNEL_LAST]]) {auto_pad = "NOTSET", ceil_mode = 0 : si64, dilations = [1, 1], kernel_shape = [2, 2], pads = [0, 0, 0, 0], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x28x28x64xf32>) -> tensor<1x14x14x64xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[POOL_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x14x14x64xf32>) -> tensor<1x64x14x14xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x14x14xf32>
}

// -----

// COM: Test GlobalAveragePool to GlobalAveragePoolChannelLast conversion
// CHECK-LABEL: func.func @test_global_avgpool_to_channel_last
func.func @test_global_avgpool_to_channel_last(%arg0: tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32> {
  %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32>
  onnx.Return %0 : tensor<1x512x1x1xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x512x7x7xf32>) -> tensor<1x7x7x512xf32>
  // CHECK: [[POOL_CHANNEL_LAST:%.+]] = "onnx.XFEGlobalAveragePool"([[INPUT_CHANNEL_LAST]]) : (tensor<1x7x7x512xf32>) -> tensor<1x1x1x512xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[POOL_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x1x1x512xf32>) -> tensor<1x512x1x1xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x512x1x1xf32>
}

// -----

// COM: Test GlobalMaxPool to GlobalMaxPoolChannelLast conversion
// CHECK-LABEL: func.func @test_global_maxpool_to_channel_last
func.func @test_global_maxpool_to_channel_last(%arg0: tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32> {
  %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32>
  onnx.Return %0 : tensor<1x512x1x1xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x512x7x7xf32>) -> tensor<1x7x7x512xf32>
  // CHECK: [[POOL_CHANNEL_LAST:%.+]] = "onnx.XFEGlobalMaxPool"([[INPUT_CHANNEL_LAST]]) : (tensor<1x7x7x512xf32>) -> tensor<1x1x1x512xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[POOL_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x1x1x512xf32>) -> tensor<1x512x1x1xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x512x1x1xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Normalization Conversion Tests
//===----------------------------------------------------------------------===//

// COM: Test InstanceNormalization to InstanceNormalizationChannelLast conversion
// CHECK-LABEL: func.func @test_instance_norm_to_channel_last
func.func @test_instance_norm_to_channel_last(%arg0: tensor<1x64x28x28xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x28x28xf32> {
  %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 1.0e-05 : f32} : (tensor<1x64x28x28xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x28x28xf32>
  onnx.Return %0 : tensor<1x64x28x28xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x28x28xf32>) -> tensor<1x28x28x64xf32>
  // CHECK: [[NORM_CHANNEL_LAST:%.+]] = "onnx.XFEInstanceNormalization"([[INPUT_CHANNEL_LAST]], %arg1, %arg2) {epsilon = 9.99999974E-6 : f32} : (tensor<1x28x28x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x28x28x64xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[NORM_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x28x28x64xf32>) -> tensor<1x64x28x28xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x28x28xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Layout Transformation Conversion Tests
//===----------------------------------------------------------------------===//

// COM: Test DepthToSpace to DepthToSpaceChannelLast conversion
// CHECK-LABEL: func.func @test_depth_to_space_to_channel_last
func.func @test_depth_to_space_to_channel_last(%arg0: tensor<1x16x4x4xf32>) -> tensor<1x4x8x8xf32> {
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64} : (tensor<1x16x4x4xf32>) -> tensor<1x4x8x8xf32>
  onnx.Return %0 : tensor<1x4x8x8xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x16x4x4xf32>) -> tensor<1x4x4x16xf32>
  // CHECK: [[D2S_CHANNEL_LAST:%.+]] = "onnx.XFEDepthToSpace"([[INPUT_CHANNEL_LAST]]) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x4x4x16xf32>) -> tensor<1x8x8x4xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[D2S_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x8x8x4xf32>) -> tensor<1x4x8x8xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x4x8x8xf32>
}

// -----

// COM: Test SpaceToDepth to SpaceToDepthChannelLast conversion
// CHECK-LABEL: func.func @test_space_to_depth_to_channel_last
func.func @test_space_to_depth_to_channel_last(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x16x4x4xf32> {
  %0 = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64} : (tensor<1x4x8x8xf32>) -> tensor<1x16x4x4xf32>
  onnx.Return %0 : tensor<1x16x4x4xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x4x8x8xf32>) -> tensor<1x8x8x4xf32>
  // CHECK: [[S2D_CHANNEL_LAST:%.+]] = "onnx.XFESpaceToDepth"([[INPUT_CHANNEL_LAST]]) {blocksize = 2 : si64} : (tensor<1x8x8x4xf32>) -> tensor<1x4x4x16xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[S2D_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x4x4x16xf32>) -> tensor<1x16x4x4xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x16x4x4xf32>
}

