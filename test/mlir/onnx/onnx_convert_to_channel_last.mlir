// RUN: onnx-mlir-opt --convert-to-channel-last --shape-inference %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test ONNX to ChannelLast Conversions
/// - Conv -> XFEConv
/// - AveragePool -> XFEAveragePool
/// - MaxPool -> XFEMaxPool
/// - GlobalAveragePool -> XFEGlobalAveragePool
/// - GlobalMaxPool -> XFEGlobalMaxPool
/// - GridSample -> XFEGridSample (rank >= 3)
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
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_OHWI]], %arg2) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x26x26x64xf32>
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
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_OHWI]], %arg2) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x28x28x64xf32>
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
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_OHWI]], %arg2) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x13x13x64xf32>
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
  // CHECK: [[CONV_CHANNEL_LAST:%.+]] = "onnx.XFEConv"([[INPUT_CHANNEL_LAST]], [[WEIGHT_ODHWI]], %arg2) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, pads = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} : (tensor<1x8x28x28x3xf32>, tensor<64x3x3x3x3xf32>, tensor<64xf32>) -> tensor<1x6x26x26x64xf32>
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
  // CHECK: [[CONVT_CHANNEL_LAST:%.+]] = "onnx.XFEConvTranspose"([[INPUT_CHANNEL_LAST]], [[WEIGHT_OHWI]], %arg2) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 4], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x28x28x3xf32>, tensor<64x4x4x3xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
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
  // CHECK: [[CONVT_CHANNEL_LAST:%.+]] = "onnx.XFEConvTranspose"([[INPUT_CHANNEL_LAST]], [[WEIGHT_OHWI]], %arg2) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], output_padding = [1, 1], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
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
  // CHECK: [[CONVT_CHANNEL_LAST:%.+]] = "onnx.XFEConvTranspose"([[INPUT_CHANNEL_LAST]], [[WEIGHT_ODHWI]], %arg2) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [2, 2, 2], pads = [0, 0, 0, 0, 0, 0], strides = [2, 2, 2]} : (tensor<1x8x28x28x3xf32>, tensor<64x2x2x2x3xf32>, tensor<64xf32>) -> tensor<1x16x56x56x64xf32>
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

// COM: Test InstanceNormalization to InstanceNormalizationChannelLast conversion (4D NCHW)
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

// COM: Test InstanceNormalization NCD (3D) conversion
// CHECK-LABEL: func.func @test_instance_norm_3d_ncd
func.func @test_instance_norm_3d_ncd(%arg0: tensor<1x32x128xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>) -> tensor<1x32x128xf32> {
  %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 1.0e-05 : f32} : (tensor<1x32x128xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x32x128xf32>
  onnx.Return %0 : tensor<1x32x128xf32>

  // CHECK: [[INPUT_NDC:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x32x128xf32>) -> tensor<1x128x32xf32>
  // CHECK: [[NORM_NDC:%.+]] = "onnx.XFEInstanceNormalization"([[INPUT_NDC]], %arg1, %arg2) {epsilon = 9.99999974E-6 : f32} : (tensor<1x128x32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x128x32xf32>
  // CHECK: [[OUTPUT_NCD:%.+]] = "onnx.Transpose"([[NORM_NDC]]) {perm = [0, 2, 1]} : (tensor<1x128x32xf32>) -> tensor<1x32x128xf32>
  // CHECK: onnx.Return [[OUTPUT_NCD]] : tensor<1x32x128xf32>
}

// -----

// COM: Test BatchNormalization NCD (3D) conversion
// CHECK-LABEL: func.func @test_batchnorm_3d_ncd
func.func @test_batchnorm_3d_ncd(%arg0: tensor<1x64x256xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x256xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.0e-05 : f32, momentum = 0.9 : f32} : (tensor<1x64x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x256xf32>
  onnx.Return %0 : tensor<1x64x256xf32>

  // CHECK: [[INPUT_NDC:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x64x256xf32>) -> tensor<1x256x64xf32>
  // CHECK: [[BN_NDC:%.+]] = "onnx.XFEBatchNormalization"([[INPUT_NDC]], %arg1, %arg2, %arg3, %arg4) {epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (tensor<1x256x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x256x64xf32>
  // CHECK: [[OUTPUT_NCD:%.+]] = "onnx.Transpose"([[BN_NDC]]) {perm = [0, 2, 1]} : (tensor<1x256x64xf32>) -> tensor<1x64x256xf32>
  // CHECK: onnx.Return [[OUTPUT_NCD]] : tensor<1x64x256xf32>
}

// -----

// COM: Test GroupNormalization 4D NCHW conversion
// CHECK-LABEL: func.func @test_groupnorm_4d_nchw
func.func @test_groupnorm_4d_nchw(%arg0: tensor<1x64x28x28xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x28x28xf32> {
  %0 = "onnx.GroupNormalization"(%arg0, %arg1, %arg2) {epsilon = 1.0e-05 : f32, num_groups = 4 : si64} : (tensor<1x64x28x28xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x28x28xf32>
  onnx.Return %0 : tensor<1x64x28x28xf32>

  // CHECK: [[INPUT_NHWC:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x28x28xf32>) -> tensor<1x28x28x64xf32>
  // CHECK: [[GN_NHWC:%.+]] = "onnx.XFEGroupNormalization"([[INPUT_NHWC]], %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, num_groups = 4 : si64} : (tensor<1x28x28x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x28x28x64xf32>
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[GN_NHWC]]) {perm = [0, 3, 1, 2]} : (tensor<1x28x28x64xf32>) -> tensor<1x64x28x28xf32>
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x28x28xf32>
}

// -----

// COM: Test GroupNormalization 3D NCD conversion
// CHECK-LABEL: func.func @test_groupnorm_3d_ncd
func.func @test_groupnorm_3d_ncd(%arg0: tensor<1x32x128xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>) -> tensor<1x32x128xf32> {
  %0 = "onnx.GroupNormalization"(%arg0, %arg1, %arg2) {epsilon = 1.0e-05 : f32, num_groups = 8 : si64} : (tensor<1x32x128xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x32x128xf32>
  onnx.Return %0 : tensor<1x32x128xf32>

  // CHECK: [[INPUT_NDC:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x32x128xf32>) -> tensor<1x128x32xf32>
  // CHECK: [[GN_NDC:%.+]] = "onnx.XFEGroupNormalization"([[INPUT_NDC]], %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, num_groups = 8 : si64} : (tensor<1x128x32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x128x32xf32>
  // CHECK: [[OUTPUT_NCD:%.+]] = "onnx.Transpose"([[GN_NDC]]) {perm = [0, 2, 1]} : (tensor<1x128x32xf32>) -> tensor<1x32x128xf32>
  // CHECK: onnx.Return [[OUTPUT_NCD]] : tensor<1x32x128xf32>
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

  // CHECK: [[SCALES_PERMUTED:%.+]] = onnx.Constant dense<[1.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00]> : tensor<4xf32>
  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32>
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

  // CHECK: [[SIZES_PERMUTED:%.+]] = onnx.Constant dense<[1, 8, 8, 3]> : tensor<4xi64>
  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32>
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

  // CHECK: [[SCALES_PERMUTED:%.+]] = onnx.Constant dense<[1.000000e+00, 5.000000e-01, 5.000000e-01, 1.000000e+00]> : tensor<4xf32>
  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x32x32xf32>) -> tensor<1x32x32x64xf32>
  // CHECK: [[RESIZE_CHANNEL_LAST:%.+]] = "onnx.XFEResize"([[INPUT_CHANNEL_LAST]], %{{.*}}, [[SCALES_PERMUTED]], %{{.*}}) {{{.*}}coordinate_transformation_mode = "asymmetric"{{.*}}mode = "nearest", nearest_mode = "floor"{{.*}}}
  // CHECK: [[OUTPUT_NCHW:%.+]] = "onnx.Transpose"([[RESIZE_CHANNEL_LAST]]) {perm = [0, 3, 1, 2]}
  // CHECK: onnx.Return [[OUTPUT_NCHW]] : tensor<1x64x16x16xf32>
}

// -----

// COM: Test Resize with axes attribute - axes must be remapped from NCHW to NHWC
// In NCHW: axes=[2,3] means resize H,W. In NHWC: axes=[1,2].
// CHECK-LABEL: func.func @test_resize_axes_remap
func.func @test_resize_axes_remap(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x8x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[2.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "onnx.Resize"(%arg0, %none, %scales, %none) {
    axes = [2, 3],
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest",
    nearest_mode = "round_prefer_floor"
  } : (tensor<1x3x4x4xf32>, none, tensor<2xf32>, none) -> tensor<1x3x8x8xf32>
  onnx.Return %0 : tensor<1x3x8x8xf32>

  // axes=[2,3] in NCHW should become axes=[1,2] in NHWC
  // CHECK: [[INPUT_CHANNEL_LAST:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
  // CHECK: "onnx.XFEResize"({{.*}}) {{{.*}}axes = [1, 2]{{.*}}}
  // CHECK: "onnx.Transpose"({{.*}}) {perm = [0, 3, 1, 2]}
}

// -----

// COM: Test Resize with Roi - roi has 2*rank=8 elements, each half permuted
// NCHW roi: [0, 0, 0.2, 0.3, 1, 1, 0.8, 0.7]
// NHWC roi: [0, 0.2, 0.3, 0, 1, 0.8, 0.7, 1]
// CHECK-LABEL: func.func @test_resize_roi_permute
func.func @test_resize_roi_permute(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x8x8xf32> {
  %roi = "onnx.Constant"() {value = dense<[0.0, 0.0, 0.2, 0.3, 1.0, 1.0, 0.8, 0.7]> : tensor<8xf32>} : () -> tensor<8xf32>
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Resize"(%arg0, %roi, %scales, %none) {
    coordinate_transformation_mode = "tf_crop_and_resize",
    mode = "nearest",
    nearest_mode = "round_prefer_floor"
  } : (tensor<1x3x4x4xf32>, tensor<8xf32>, tensor<4xf32>, none) -> tensor<1x3x8x8xf32>
  onnx.Return %0 : tensor<1x3x8x8xf32>

  // Roi [0,0,0.2,0.3, 1,1,0.8,0.7] permuted per-half with [0,2,3,1]:
  // -> [0, 0.2, 0.3, 0, 1, 0.8, 0.7, 1]
  // CHECK-DAG: [[SCALES_PERMUTED:%.+]] = onnx.Constant dense<[1.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00]> : tensor<4xf32>
  // CHECK-DAG: [[ROI_PERMUTED:%.+]] = onnx.Constant dense<[0.000000e+00, 2.000000e-01, 3.000000e-01, 0.000000e+00, 1.000000e+00, 8.000000e-01, 0.699999988, 1.000000e+00]> : tensor<8xf32>
  // CHECK: "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
  // CHECK: "onnx.XFEResize"
  // CHECK: "onnx.Transpose"({{.*}}) {perm = [0, 3, 1, 2]}
}

// -----

//===----------------------------------------------------------------------===//
/// Quantized Type Tests - Per-tensor quantization
//===----------------------------------------------------------------------===//

// COM: Test Conv with per-tensor quantized types
// CHECK-LABEL: func.func @test_conv_quantized_per_tensor
func.func @test_conv_quantized_per_tensor(
    %arg0: tensor<1x3x28x28x!quant.uniform<u8:f32, 0.05:128>>,
    %arg1: tensor<64x3x3x3x!quant.uniform<i8:f32, 0.02>>,
    %arg2: tensor<64x!quant.uniform<i8:f32, 0.01>>)
    -> tensor<1x64x26x26x!quant.uniform<u8:f32, 0.1:128>> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
    dilations = [1, 1],
    group = 1 : si64,
    pads = [0, 0, 0, 0],
    strides = [1, 1]
  } : (tensor<1x3x28x28x!quant.uniform<u8:f32, 0.05:128>>,
       tensor<64x3x3x3x!quant.uniform<i8:f32, 0.02>>,
       tensor<64x!quant.uniform<i8:f32, 0.01>>)
    -> tensor<1x64x26x26x!quant.uniform<u8:f32, 0.1:128>>
  onnx.Return %0 : tensor<1x64x26x26x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK: "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
  // CHECK: "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]}
  // CHECK: "onnx.XFEConv"
  // CHECK: "onnx.Transpose"({{.*}}) {perm = [0, 3, 1, 2]}
}

// -----

//===----------------------------------------------------------------------===//
/// Quantized Type Tests - Per-axis quantization with axis remapping
//===----------------------------------------------------------------------===//

// COM: Test Conv with per-axis quantized weight (axis=0 in NCHW stays axis=0 in OHWI)
// CHECK-LABEL: func.func @test_conv_quantized_per_axis
func.func @test_conv_quantized_per_axis(
    %arg0: tensor<1x3x8x8x!quant.uniform<u8:f32, 0.05:128>>,
    %arg1: tensor<16x3x3x3x!quant.uniform<i8:f32:0, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16}>>,
    %arg2: tensor<16xf32>)
    -> tensor<1x16x6x6x!quant.uniform<u8:f32, 0.1:128>> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
    dilations = [1, 1],
    group = 1 : si64,
    pads = [0, 0, 0, 0],
    strides = [1, 1]
  } : (tensor<1x3x8x8x!quant.uniform<u8:f32, 0.05:128>>,
       tensor<16x3x3x3x!quant.uniform<i8:f32:0, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16}>>,
       tensor<16xf32>)
    -> tensor<1x16x6x6x!quant.uniform<u8:f32, 0.1:128>>
  onnx.Return %0 : tensor<1x16x6x6x!quant.uniform<u8:f32, 0.1:128>>

  // Weight transpose [0,2,3,1] keeps axis=0 as axis=0
  // CHECK: "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
  // CHECK: "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]}
  // CHECK: "onnx.XFEConv"
  // Output per-axis quant on axis=1 (channel in NCHW) remaps to axis=3 (channel in NHWC)
  // CHECK: "onnx.Transpose"({{.*}}) {perm = [0, 3, 1, 2]}
}

// -----

// COM: Test AveragePool with quantized types
// CHECK-LABEL: func.func @test_avgpool_quantized
func.func @test_avgpool_quantized(
    %arg0: tensor<1x64x28x28x!quant.uniform<u8:f32, 0.05:128>>)
    -> tensor<1x64x14x14x!quant.uniform<u8:f32, 0.05:128>> {
  %0 = "onnx.AveragePool"(%arg0) {
    kernel_shape = [2, 2],
    strides = [2, 2],
    pads = [0, 0, 0, 0]
  } : (tensor<1x64x28x28x!quant.uniform<u8:f32, 0.05:128>>)
    -> tensor<1x64x14x14x!quant.uniform<u8:f32, 0.05:128>>
  onnx.Return %0 : tensor<1x64x14x14x!quant.uniform<u8:f32, 0.05:128>>

  // CHECK: "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
  // CHECK: "onnx.XFEAveragePool"
  // CHECK: "onnx.Transpose"({{.*}}) {perm = [0, 3, 1, 2]}
}

// -----

//===----------------------------------------------------------------------===//
/// GridSample → XFEGridSample (3D / rank-3 NCL with channel-last sandwich)
//===----------------------------------------------------------------------===//

// COM: 1D spatial GridSample: NCHW layout (N,C,L); grid (N,L_out,1).
// CHECK-LABEL: func.func @test_gridsample_rank3_to_xfe_channel_last
func.func @test_gridsample_rank3_to_xfe_channel_last(
    %arg0: tensor<1x3x8xf32>, %arg1: tensor<1x4x1xf32>) -> tensor<1x3x4xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {
    align_corners = 0 : si64,
    mode = "linear",
    padding_mode = "zeros"
  } : (tensor<1x3x8xf32>, tensor<1x4x1xf32>) -> tensor<1x3x4xf32>
  onnx.Return %0 : tensor<1x3x4xf32>

  // CHECK: [[IN:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]}
  // CHECK: "onnx.XFEGridSample"([[IN]], %arg1)
  // CHECK: "onnx.Transpose"({{.*}}) {perm = [0, 2, 1]}
}

// -----

//===----------------------------------------------------------------------===//
/// GridSample → XFEGridSample (4D NCHW with NHWC sandwich)
//===----------------------------------------------------------------------===//

// COM: Test GridSample to XFEGridSample with explicit input/output transposes
// CHECK-LABEL: func.func @test_gridsample_to_xfe_channel_last
func.func @test_gridsample_to_xfe_channel_last(%arg0: tensor<1x3x4x4xf32>, %arg1: tensor<1x2x2x2xf32>) -> tensor<1x3x2x2xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {
    align_corners = 0 : si64,
    mode = "linear",
    padding_mode = "zeros"
  } : (tensor<1x3x4x4xf32>, tensor<1x2x2x2xf32>) -> tensor<1x3x2x2xf32>
  onnx.Return %0 : tensor<1x3x2x2xf32>

  // CHECK: [[IN_NHWC:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32>
  // CHECK: [[GS:%.+]] = "onnx.XFEGridSample"([[IN_NHWC]], %arg1) {{{.*}}align_corners = 0 : si64{{.*}}mode = "linear"{{.*}}padding_mode = "zeros"{{.*}}} : (tensor<1x4x4x3xf32>, tensor<1x2x2x2xf32>) -> tensor<1x2x2x3xf32>
  // CHECK: [[OUT:%.+]] = "onnx.Transpose"([[GS]]) {perm = [0, 3, 1, 2]} : (tensor<1x2x2x3xf32>) -> tensor<1x3x2x2xf32>
  // CHECK: onnx.Return [[OUT]] : tensor<1x3x2x2xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// GridSample → XFEGridSample (5D NCDHW with NDHWC sandwich)
//===----------------------------------------------------------------------===//

// COM: Volumetric GridSample uses rank-5 X and grid; channel-last is NDHWC.
// CHECK-LABEL: func.func @test_gridsample_5d_to_xfe_channel_last
func.func @test_gridsample_5d_to_xfe_channel_last(
    %arg0: tensor<1x2x3x4x5xf32>,
    %arg1: tensor<1x2x3x4x3xf32>) -> tensor<1x2x3x4x5xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {
    align_corners = 0 : si64,
    mode = "linear",
    padding_mode = "zeros"
  } : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<1x2x3x4x5xf32>
  onnx.Return %0 : tensor<1x2x3x4x5xf32>

  // CHECK: [[IN:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 4, 1]}
  // CHECK: "onnx.XFEGridSample"([[IN]], %arg1)
  // CHECK: "onnx.Transpose"({{.*}}) {perm = [0, 4, 1, 2, 3]}
}

// -----

//===----------------------------------------------------------------------===//
/// GridSample → XFEGridSample (4D per-tensor quant)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_gridsample_quant_per_tensor
func.func @test_gridsample_quant_per_tensor(
    %arg0: tensor<1x3x4x4x!quant.uniform<u8:f32, 0.05:128>>,
    %arg1: tensor<1x2x2x2xf32>) -> tensor<1x3x2x2x!quant.uniform<u8:f32, 0.1:128>> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {
    align_corners = 0 : si64,
    mode = "linear",
    padding_mode = "zeros"
  } : (tensor<1x3x4x4x!quant.uniform<u8:f32, 0.05:128>>, tensor<1x2x2x2xf32>)
    -> tensor<1x3x2x2x!quant.uniform<u8:f32, 0.1:128>>
  onnx.Return %0 : tensor<1x3x2x2x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK: [[IN_NHWC:%.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4x!quant.uniform<u8:f32, 5.000000e-02:128>>) -> tensor<1x4x4x3x!quant.uniform<u8:f32, 5.000000e-02:128>>
  // CHECK: [[GS:%.+]] = "onnx.XFEGridSample"([[IN_NHWC]], %arg1) {{{.*}}align_corners = 0 : si64{{.*}}mode = "linear"{{.*}}padding_mode = "zeros"{{.*}}} : (tensor<1x4x4x3x!quant.uniform<u8:f32, 5.000000e-02:128>>, tensor<1x2x2x2xf32>) -> tensor<1x2x2x3x!quant.uniform<u8:f32, 1.000000e-01:128>>
  // CHECK: [[OUT:%.+]] = "onnx.Transpose"([[GS]]) {perm = [0, 3, 1, 2]} : (tensor<1x2x2x3x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x3x2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  // CHECK: onnx.Return [[OUT]] : tensor<1x3x2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

