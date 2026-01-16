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
  // CHECK: [[WEIGHT_OHWI:%.+]] = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xf32>
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_OHWI]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x26x26x64xf32>
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
  // CHECK: [[WEIGHT_OHWI:%.+]] = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xf32>
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_OHWI]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x28x28x64xf32>
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
  // CHECK: [[WEIGHT_OHWI:%.+]] = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xf32>
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_OHWI]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x13x13x64xf32>
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
  // CHECK: [[WEIGHT_ODHWI:%.+]] = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 4, 1]} : (tensor<64x3x3x3x3xf32>) -> tensor<64x3x3x3x3xf32>
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_ODHWI]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, pads = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} : (tensor<1x8x28x28x3xf32>, tensor<64x3x3x3x3xf32>, tensor<64xf32>) -> tensor<1x6x26x26x64xf32>
  // CHECK: [[OUTPUT_NCDHW:%.+]] = "onnx.Transpose"([[CONV_CHANNEL_LAST]]) {perm = [0, 4, 1, 2, 3]} : (tensor<1x6x26x26x64xf32>) -> tensor<1x64x6x26x26xf32>
  // CHECK: onnx.Return [[OUTPUT_NCDHW]] : tensor<1x64x6x26x26xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// ConvTranspose Conversion Tests
//===----------------------------------------------------------------------===//

// COM: Test basic 2D transposed convolution conversion
// CHECK-LABEL: func.func @test_convtranspose_to_xfe_conv_transpose_channel_last
func.func @test_convtranspose_to_xfe_conv_transpose_channel_last(%arg0: tensor<1x3x28x28xf32>, %arg1: tensor<3x64x4x4xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %arg2) {
    dilations = [1, 1],
    group = 1 : si64,
    kernel_shape = [4, 4],
    pads = [1, 1, 1, 1],
    strides = [2, 2]
  } : (tensor<1x3x28x28xf32>, tensor<3x64x4x4xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
  onnx.Return %0 : tensor<1x64x56x56xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x28x28xf32>) -> tensor<1x28x28x3xf32>
  // CHECK: [[WEIGHT_OHWI:%.+]] = "onnx.Transpose"(%arg1) {perm = [1, 2, 3, 0]} : (tensor<3x64x4x4xf32>) -> tensor<64x4x4x3xf32>
  // CHECK: [[CONVT_CHANNEL_LAST:%.+]] = "onnx.XFEConvTranspose"([[INPUT_CHANNEL_LAST]], [[WEIGHT_OHWI]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 4], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x28x28x3xf32>, tensor<64x4x4x3xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[CONVT_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x64xf32>) -> tensor<1x64x56x56xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x56x56xf32>
}

// -----

// COM: Test transposed convolution with output_padding
// CHECK-LABEL: func.func @test_convtranspose_to_xfe_conv_transpose_output_padding
func.func @test_convtranspose_to_xfe_conv_transpose_output_padding(%arg0: tensor<1x3x28x28xf32>, %arg1: tensor<3x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %arg2) {
    dilations = [1, 1],
    group = 1 : si64,
    kernel_shape = [3, 3],
    output_padding = [1, 1],
    pads = [1, 1, 1, 1],
    strides = [2, 2]
  } : (tensor<1x3x28x28xf32>, tensor<3x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
  onnx.Return %0 : tensor<1x64x56x56xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x28x28xf32>) -> tensor<1x28x28x3xf32>
  // CHECK: [[WEIGHT_OHWI:%.+]] = "onnx.Transpose"(%arg1) {perm = [1, 2, 3, 0]} : (tensor<3x64x3x3xf32>) -> tensor<64x3x3x3xf32>
  // CHECK: [[CONVT_CHANNEL_LAST:%.+]] = "onnx.XFEConvTranspose"([[INPUT_CHANNEL_LAST]], [[WEIGHT_OHWI]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], output_padding = [1, 1], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[CONVT_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x64xf32>) -> tensor<1x64x56x56xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x56x56xf32>
}

// -----

// COM: Test 3D transposed convolution conversion (5D tensors)
// CHECK-LABEL: func.func @test_convtranspose3d_to_xfe_conv_transpose_channel_last
func.func @test_convtranspose3d_to_xfe_conv_transpose_channel_last(%arg0: tensor<1x3x8x28x28xf32>, %arg1: tensor<3x64x2x2x2xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x16x56x56xf32> {
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %arg2) {
    dilations = [1, 1, 1],
    group = 1 : si64,
    kernel_shape = [2, 2, 2],
    pads = [0, 0, 0, 0, 0, 0],
    strides = [2, 2, 2]
  } : (tensor<1x3x8x28x28xf32>, tensor<3x64x2x2x2xf32>, tensor<64xf32>) -> tensor<1x64x16x56x56xf32>
  onnx.Return %0 : tensor<1x64x16x56x56xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 4, 1]} : (tensor<1x3x8x28x28xf32>) -> tensor<1x8x28x28x3xf32>
  // CHECK: [[WEIGHT_ODHWI:%.+]] = "onnx.Transpose"(%arg1) {perm = [1, 2, 3, 4, 0]} : (tensor<3x64x2x2x2xf32>) -> tensor<64x2x2x2x3xf32>
  // CHECK: [[CONVT_CHANNEL_LAST:%.+]] = "onnx.XFEConvTranspose"([[INPUT_CHANNEL_LAST]], [[WEIGHT_ODHWI]], %arg2) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [2, 2, 2], pads = [0, 0, 0, 0, 0, 0], strides = [2, 2, 2]} : (tensor<1x8x28x28x3xf32>, tensor<64x2x2x2x3xf32>, tensor<64xf32>) -> tensor<1x16x56x56x64xf32>
  // CHECK: [[OUTPUT_NCDHW:%.+]] = "onnx.Transpose"([[CONVT_CHANNEL_LAST]]) {perm = [0, 4, 1, 2, 3]} : (tensor<1x16x56x56x64xf32>) -> tensor<1x64x16x56x56xf32>
  // CHECK: onnx.Return [[OUTPUT_NCDHW]] : tensor<1x64x16x56x56xf32>
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

// -----

//===----------------------------------------------------------------------===//
/// Resize Conversion Tests
//===----------------------------------------------------------------------===//

// COM: Test Resize with scales to XFEResize conversion
// CHECK-LABEL: func.func @test_resize_scales_to_channel_last
func.func @test_resize_scales_to_channel_last(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x8x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.Resize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest",
    nearest_mode = "round_prefer_floor"
  } : (tensor<1x3x4x4xf32>, none, tensor<4xf32>, none) -> tensor<1x3x8x8xf32>
  onnx.Return %0 : tensor<1x3x8x8xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32>
  // CHECK: [[SCALES_PERMUTED:%.+]] = "onnx.Gather"
  // CHECK: [[RESIZE_CHANNEL_LAST:%.+]] = "onnx.XFEResize"([[INPUT_CHANNEL_LAST]], %{{.*}}, [[SCALES_PERMUTED]], %{{.*}}) {{{.*}}coordinate_transformation_mode = "half_pixel"{{.*}}mode = "nearest", nearest_mode = "round_prefer_floor"{{.*}}}
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[RESIZE_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]}
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x3x8x8xf32>
}

// -----

// COM: Test Resize with sizes to XFEResize conversion
// CHECK-LABEL: func.func @test_resize_sizes_to_channel_last
func.func @test_resize_sizes_to_channel_last(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x8x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = "onnx.Constant"() {value = dense<[1, 3, 8, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
  %0 = "onnx.Resize"(%arg0, %none, %none, %sizes) {
    coordinate_transformation_mode = "half_pixel",
    mode = "linear"
  } : (tensor<1x3x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x3x8x8xf32>
  onnx.Return %0 : tensor<1x3x8x8xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32>
  // CHECK: [[SIZES_PERMUTED:%.+]] = "onnx.Gather"
  // CHECK: [[RESIZE_CHANNEL_LAST:%.+]] = "onnx.XFEResize"([[INPUT_CHANNEL_LAST]], %{{.*}}, %{{.*}}, [[SIZES_PERMUTED]]) {{{.*}}coordinate_transformation_mode = "half_pixel"{{.*}}mode = "linear"{{.*}}}
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[RESIZE_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]}
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x3x8x8xf32>
}

// -----

// COM: Test Resize downsample to XFEResize conversion
// CHECK-LABEL: func.func @test_resize_downsample_to_channel_last
func.func @test_resize_downsample_to_channel_last(%arg0: tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 0.5, 0.5]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.Resize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "asymmetric",
    mode = "nearest",
    nearest_mode = "floor"
  } : (tensor<1x64x32x32xf32>, none, tensor<4xf32>, none) -> tensor<1x64x16x16xf32>
  onnx.Return %0 : tensor<1x64x16x16xf32>

  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x32x32xf32>) -> tensor<1x32x32x64xf32>
  // CHECK: [[SCALES_PERMUTED:%.+]] = "onnx.Gather"
  // CHECK: [[RESIZE_CHANNEL_LAST:%.+]] = "onnx.XFEResize"([[INPUT_CHANNEL_LAST]], %{{.*}}, [[SCALES_PERMUTED]], %{{.*}}) {{{.*}}coordinate_transformation_mode = "asymmetric"{{.*}}mode = "nearest", nearest_mode = "floor"{{.*}}}
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[RESIZE_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]}
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x16x16xf32>
}

