// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Shape inference tests for XFE Operations  
/// Domain: com.amd.xfe
//===----------------------------------------------------------------------===//

// -----

//===----------------------------------------------------------------------===//
/// XFE MatMulBias Tests
//===----------------------------------------------------------------------===//

// COM: Test basic 2D x 2D matrix multiplication
func.func @test_xfe_matmulbias_2d(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMatMulBias"(%arg0, %arg1, %arg2) : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_matmulbias_2d
  // CHECK: [[RES:%.+]] = "onnx.XFEMatMulBias"(%arg0, %arg1, %arg2) : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  // CHECK: onnx.Return [[RES]] : tensor<4x16xf32>
}

// -----

// COM: Test 3D (batched) matrix multiplication
func.func @test_xfe_matmulbias_3d(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>, %arg2: tensor<16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMatMulBias"(%arg0, %arg1, %arg2) : (tensor<2x4x8xf32>, tensor<2x8x16xf32>, tensor<16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_matmulbias_3d
  // CHECK: [[RES:%.+]] = "onnx.XFEMatMulBias"(%arg0, %arg1, %arg2) : (tensor<2x4x8xf32>, tensor<2x8x16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x4x16xf32>
}

// -----

// COM: Test with dynamic dimensions
func.func @test_xfe_matmulbias_dynamic(%arg0: tensor<?x8xf32>, %arg1: tensor<8x?xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMatMulBias"(%arg0, %arg1, %arg2) : (tensor<?x8xf32>, tensor<8x?xf32>, tensor<?xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_matmulbias_dynamic
  // CHECK: [[RES:%.+]] = "onnx.XFEMatMulBias"(%arg0, %arg1, %arg2) : (tensor<?x8xf32>, tensor<8x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  // CHECK: onnx.Return [[RES]] : tensor<?x?xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE ConvChannelLast Tests
//===----------------------------------------------------------------------===//

// COM: Test basic channel-last convolution
func.func @test_xfe_conv_channel_last_basic(%arg0: tensor<1x28x28x3xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%arg0, %arg1, %arg2) {
    strides = [1, 1],
    pads = [0, 0, 0, 0],
    dilations = [1, 1]
  } : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_conv_channel_last_basic
  // CHECK: [[RES:%.+]] = "onnx.XFEConv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x26x26x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x26x26x64xf32>
}

// -----

// COM: Test channel-last convolution with padding
func.func @test_xfe_conv_channel_last_padded(%arg0: tensor<1x28x28x3xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%arg0, %arg1, %arg2) {
    strides = [1, 1],
    pads = [1, 1, 1, 1],
    dilations = [1, 1]
  } : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_conv_channel_last_padded
  // CHECK: [[RES:%.+]] = "onnx.XFEConv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x28x28x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x28x28x64xf32>
}

// -----

// COM: Test channel-last convolution with stride=2
func.func @test_xfe_conv_channel_last_strided(%arg0: tensor<1x28x28x3xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%arg0, %arg1, %arg2) {
    strides = [2, 2],
    pads = [0, 0, 0, 0],
    dilations = [1, 1]
  } : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_conv_channel_last_strided
  // CHECK: [[RES:%.+]] = "onnx.XFEConv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x13x13x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x13x13x64xf32>
}

// -----

// COM: Test 3D (1D spatial) convolution
func.func @test_xfe_conv_channel_last_1d(%arg0: tensor<2x128x16xf32>, %arg1: tensor<32x5x16xf32>, %arg2: tensor<32xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%arg0, %arg1, %arg2) {
    strides = [1],
    pads = [0, 0],
    dilations = [1]
  } : (tensor<2x128x16xf32>, tensor<32x5x16xf32>, tensor<32xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_conv_channel_last_1d
  // CHECK: [[RES:%.+]] = "onnx.XFEConv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, pads = [0, 0], strides = [1]} : (tensor<2x128x16xf32>, tensor<32x5x16xf32>, tensor<32xf32>) -> tensor<2x124x32xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x124x32xf32>
}

// -----

// COM: Test 5D (3D spatial) convolution
func.func @test_xfe_conv_channel_last_3d(%arg0: tensor<1x8x16x16x3xf32>, %arg1: tensor<64x3x3x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%arg0, %arg1, %arg2) {
    strides = [1, 1, 1],
    pads = [0, 0, 0, 0, 0, 0],
    dilations = [1, 1, 1]
  } : (tensor<1x8x16x16x3xf32>, tensor<64x3x3x3x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_conv_channel_last_3d
  // CHECK: [[RES:%.+]] = "onnx.XFEConv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, pads = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} : (tensor<1x8x16x16x3xf32>, tensor<64x3x3x3x3xf32>, tensor<64xf32>) -> tensor<1x6x14x14x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x6x14x14x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE ConvTranspose Tests (channel-last layout)
//===----------------------------------------------------------------------===//

// COM: Test basic 2D transposed convolution with stride=2
func.func @test_xfe_convtranspose_channel_last_basic(%arg0: tensor<1x28x28x3xf32>, %arg1: tensor<64x4x4x3xf32>, %arg2: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConvTranspose"(%arg0, %arg1, %arg2) {
    strides = [2, 2],
    pads = [1, 1, 1, 1],
    dilations = [1, 1],
    kernel_shape = [4, 4]
  } : (tensor<1x28x28x3xf32>, tensor<64x4x4x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_convtranspose_channel_last_basic
  // CHECK: [[RES:%.+]] = "onnx.XFEConvTranspose"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 4], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x28x28x3xf32>, tensor<64x4x4x3xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x56x56x64xf32>
}

// -----

// COM: Test transposed convolution with output_padding
func.func @test_xfe_convtranspose_channel_last_output_padding(%arg0: tensor<1x28x28x3xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConvTranspose"(%arg0, %arg1, %arg2) {
    strides = [2, 2],
    pads = [1, 1, 1, 1],
    dilations = [1, 1],
    kernel_shape = [3, 3],
    output_padding = [1, 1]
  } : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_convtranspose_channel_last_output_padding
  // CHECK: [[RES:%.+]] = "onnx.XFEConvTranspose"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], output_padding = [1, 1], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x56x56x64xf32>
}

// -----

// COM: Test 1D transposed convolution
func.func @test_xfe_convtranspose_channel_last_1d(%arg0: tensor<2x28x16xf32>, %arg1: tensor<32x4x16xf32>, %arg2: tensor<32xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConvTranspose"(%arg0, %arg1, %arg2) {
    strides = [2],
    pads = [0, 0],
    dilations = [1],
    kernel_shape = [4]
  } : (tensor<2x28x16xf32>, tensor<32x4x16xf32>, tensor<32xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_convtranspose_channel_last_1d
  // CHECK: [[RES:%.+]] = "onnx.XFEConvTranspose"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [4], pads = [0, 0], strides = [2]} : (tensor<2x28x16xf32>, tensor<32x4x16xf32>, tensor<32xf32>) -> tensor<2x58x32xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x58x32xf32>
}

// -----

// COM: Test 3D transposed convolution
func.func @test_xfe_convtranspose_channel_last_3d(%arg0: tensor<1x8x16x16x3xf32>, %arg1: tensor<64x2x2x2x3xf32>, %arg2: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConvTranspose"(%arg0, %arg1, %arg2) {
    strides = [2, 2, 2],
    pads = [0, 0, 0, 0, 0, 0],
    dilations = [1, 1, 1],
    kernel_shape = [2, 2, 2]
  } : (tensor<1x8x16x16x3xf32>, tensor<64x2x2x2x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_convtranspose_channel_last_3d
  // CHECK: [[RES:%.+]] = "onnx.XFEConvTranspose"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [2, 2, 2], pads = [0, 0, 0, 0, 0, 0], strides = [2, 2, 2]} : (tensor<1x8x16x16x3xf32>, tensor<64x2x2x2x3xf32>, tensor<64xf32>) -> tensor<1x16x32x32x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x16x32x32x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE Pooling Tests (channel-last layout)
//===----------------------------------------------------------------------===//

// COM: Test AveragePoolChannelLast
func.func @test_xfe_avgpool_channel_last(%arg0: tensor<1x28x28x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEAveragePool"(%arg0) {
    kernel_shape = [2, 2],
    strides = [2, 2],
    pads = [0, 0, 0, 0]
  } : (tensor<1x28x28x64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_avgpool_channel_last
  // CHECK: [[RES:%.+]] = "onnx.XFEAveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x28x28x64xf32>) -> tensor<1x14x14x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x14x14x64xf32>
}

// -----

// COM: Test MaxPoolChannelLast
func.func @test_xfe_maxpool_channel_last(%arg0: tensor<1x28x28x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMaxPool"(%arg0) {
    kernel_shape = [2, 2],
    strides = [2, 2],
    pads = [0, 0, 0, 0],
    dilations = [1, 1]
  } : (tensor<1x28x28x64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_maxpool_channel_last
  // CHECK: [[RES:%.+]] = "onnx.XFEMaxPool"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x28x28x64xf32>) -> tensor<1x14x14x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x14x14x64xf32>
}

// -----

// COM: Test GlobalAveragePoolChannelLast
func.func @test_xfe_global_avgpool_channel_last(%arg0: tensor<1x7x7x512xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalAveragePool"(%arg0) : (tensor<1x7x7x512xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_global_avgpool_channel_last
  // CHECK: [[RES:%.+]] = "onnx.XFEGlobalAveragePool"(%arg0) : (tensor<1x7x7x512xf32>) -> tensor<1x1x1x512xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x1x1x512xf32>
}

// -----

// COM: Test GlobalMaxPoolChannelLast
func.func @test_xfe_global_maxpool_channel_last(%arg0: tensor<1x7x7x512xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalMaxPool"(%arg0) : (tensor<1x7x7x512xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_global_maxpool_channel_last
  // CHECK: [[RES:%.+]] = "onnx.XFEGlobalMaxPool"(%arg0) : (tensor<1x7x7x512xf32>) -> tensor<1x1x1x512xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x1x1x512xf32>
}

// -----

// COM: Test 1D AveragePoolChannelLast
func.func @test_xfe_avgpool_channel_last_1d(%arg0: tensor<2x128x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEAveragePool"(%arg0) {
    kernel_shape = [3],
    strides = [2],
    pads = [0, 0]
  } : (tensor<2x128x32xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_avgpool_channel_last_1d
  // CHECK: [[RES:%.+]] = "onnx.XFEAveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [3], pads = [0, 0], strides = [2]} : (tensor<2x128x32xf32>) -> tensor<2x63x32xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x63x32xf32>
}

// -----

// COM: Test 3D MaxPoolChannelLast
func.func @test_xfe_maxpool_channel_last_3d(%arg0: tensor<1x8x16x16x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMaxPool"(%arg0) {
    kernel_shape = [2, 2, 2],
    strides = [2, 2, 2],
    pads = [0, 0, 0, 0, 0, 0],
    dilations = [1, 1, 1]
  } : (tensor<1x8x16x16x64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_maxpool_channel_last_3d
  // CHECK: [[RES:%.+]] = "onnx.XFEMaxPool"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1, 1], kernel_shape = [2, 2, 2], pads = [0, 0, 0, 0, 0, 0], strides = [2, 2, 2]} : (tensor<1x8x16x16x64xf32>) -> tensor<1x4x8x8x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x4x8x8x64xf32>
}

// -----

// COM: Test 1D GlobalAveragePoolChannelLast
func.func @test_xfe_global_avgpool_channel_last_1d(%arg0: tensor<2x256x128xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalAveragePool"(%arg0) : (tensor<2x256x128xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_global_avgpool_channel_last_1d
  // CHECK: [[RES:%.+]] = "onnx.XFEGlobalAveragePool"(%arg0) : (tensor<2x256x128xf32>) -> tensor<2x1x128xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x1x128xf32>
}

// -----

// COM: Test 3D GlobalMaxPoolChannelLast
func.func @test_xfe_global_maxpool_channel_last_3d(%arg0: tensor<1x8x16x16x512xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalMaxPool"(%arg0) : (tensor<1x8x16x16x512xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_global_maxpool_channel_last_3d
  // CHECK: [[RES:%.+]] = "onnx.XFEGlobalMaxPool"(%arg0) : (tensor<1x8x16x16x512xf32>) -> tensor<1x1x1x1x512xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x1x1x1x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE Normalization Tests
//===----------------------------------------------------------------------===//

// COM: Test InstanceNormalizationChannelLast (preserves input shape)
func.func @test_xfe_instance_norm_channel_last(%arg0: tensor<1x28x28x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEInstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 1.0e-05 : f32} : (tensor<1x28x28x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_instance_norm_channel_last
  // CHECK: [[RES:%.+]] = "onnx.XFEInstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32} : (tensor<1x28x28x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x28x28x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x28x28x64xf32>
}

// -----

// COM: Test 1D InstanceNormalizationChannelLast (preserves shape)
func.func @test_xfe_instance_norm_channel_last_1d(%arg0: tensor<2x256x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEInstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 1.0e-05 : f32} : (tensor<2x256x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_instance_norm_channel_last_1d
  // CHECK: [[RES:%.+]] = "onnx.XFEInstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32} : (tensor<2x256x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<2x256x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x256x64xf32>
}

// -----

// COM: Test 3D InstanceNormalizationChannelLast (preserves shape)
func.func @test_xfe_instance_norm_channel_last_3d(%arg0: tensor<1x8x16x16x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEInstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 1.0e-05 : f32} : (tensor<1x8x16x16x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_instance_norm_channel_last_3d
  // CHECK: [[RES:%.+]] = "onnx.XFEInstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32} : (tensor<1x8x16x16x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x8x16x16x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x8x16x16x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE Layout Transformation Tests
//===----------------------------------------------------------------------===//

// COM: Test DepthToSpaceChannelLast (increases spatial, decreases channels)
func.func @test_xfe_depth_to_space(%arg0: tensor<1x4x4x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%arg0) {blocksize = 2 : si64} : (tensor<1x4x4x16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_depth_to_space
  // CHECK: [[RES:%.+]] = "onnx.XFEDepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x4x4x16xf32>) -> tensor<1x8x8x4xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x8x8x4xf32>
}

// -----

// COM: Test SpaceToDepthChannelLast (decreases spatial, increases channels)
func.func @test_xfe_space_to_depth(%arg0: tensor<1x8x8x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFESpaceToDepth"(%arg0) {blocksize = 2 : si64} : (tensor<1x8x8x4xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_space_to_depth
  // CHECK: [[RES:%.+]] = "onnx.XFESpaceToDepth"(%arg0) {blocksize = 2 : si64} : (tensor<1x8x8x4xf32>) -> tensor<1x4x4x16xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x4x4x16xf32>
}

// -----

// COM: Test DepthToSpace and SpaceToDepth are inverse operations
func.func @test_xfe_depth_space_roundtrip(%arg0: tensor<1x4x4x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%arg0) {blocksize = 2 : si64} : (tensor<1x4x4x16xf32>) -> tensor<*xf32>
  %1 = "onnx.XFESpaceToDepth"(%0) {blocksize = 2 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %1 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_depth_space_roundtrip
  // CHECK: [[D2S:%.+]] = "onnx.XFEDepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x4x4x16xf32>) -> tensor<1x8x8x4xf32>
  // CHECK: [[S2D:%.+]] = "onnx.XFESpaceToDepth"([[D2S]]) {blocksize = 2 : si64} : (tensor<1x8x8x4xf32>) -> tensor<1x4x4x16xf32>
  // CHECK: onnx.Return [[S2D]] : tensor<1x4x4x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE Resize Tests (channel-last layout)
//===----------------------------------------------------------------------===//

// COM: Test ResizeChannelLast with scales (2x upscale)
func.func @test_xfe_resize_scales_upsample(%arg0: tensor<1x4x4x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest",
    nearest_mode = "round_prefer_floor"
  } : (tensor<1x4x4x3xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_resize_scales_upsample
  // CHECK: [[RES:%.+]] = "onnx.XFEResize"(%arg0, %{{.*}}, %{{.*}}, %{{.*}}) {{{.*}}coordinate_transformation_mode = "half_pixel"{{.*}}mode = "nearest"{{.*}}nearest_mode = "round_prefer_floor"{{.*}}} : (tensor<1x4x4x3xf32>, none, tensor<4xf32>, none) -> tensor<1x8x8x3xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x8x8x3xf32>
}

// -----

// COM: Test ResizeChannelLast with scales (downsample)
func.func @test_xfe_resize_scales_downsample(%arg0: tensor<1x32x32x64xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 0.5, 0.5, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "asymmetric",
    mode = "nearest",
    nearest_mode = "floor"
  } : (tensor<1x32x32x64xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_resize_scales_downsample
  // CHECK: [[RES:%.+]] = "onnx.XFEResize"(%arg0, %{{.*}}, %{{.*}}, %{{.*}}) {{{.*}}coordinate_transformation_mode = "asymmetric"{{.*}}mode = "nearest"{{.*}}nearest_mode = "floor"{{.*}}} : (tensor<1x32x32x64xf32>, none, tensor<4xf32>, none) -> tensor<1x16x16x64xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x16x16x64xf32>
}

// -----

// COM: Test ResizeChannelLast with sizes
func.func @test_xfe_resize_sizes(%arg0: tensor<1x4x4x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = "onnx.Constant"() {value = dense<[1, 8, 8, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  %0 = "onnx.XFEResize"(%arg0, %none, %none, %sizes) {
    coordinate_transformation_mode = "half_pixel",
    mode = "linear"
  } : (tensor<1x4x4x3xf32>, none, none, tensor<4xi64>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_resize_sizes
  // CHECK: [[RES:%.+]] = "onnx.XFEResize"(%arg0, %{{.*}}, %{{.*}}, %{{.*}}) {{{.*}}coordinate_transformation_mode = "half_pixel"{{.*}}mode = "linear"{{.*}}} : (tensor<1x4x4x3xf32>, none, none, tensor<4xi64>) -> tensor<1x8x8x3xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x8x8x3xf32>
}

// -----

// COM: Test ResizeChannelLast with dynamic scales (output should be dynamic)
func.func @test_xfe_resize_dynamic_scales(%arg0: tensor<1x4x4x3xf32>, %arg1: tensor<4xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XFEResize"(%arg0, %none, %arg1, %none) {
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest"
  } : (tensor<1x4x4x3xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_resize_dynamic_scales
  // CHECK: [[RES:%.+]] = "onnx.XFEResize"(%arg0, %{{.*}}, %arg1, %{{.*}}) {{{.*}}coordinate_transformation_mode = "half_pixel"{{.*}}mode = "nearest"{{.*}}} : (tensor<1x4x4x3xf32>, none, tensor<4xf32>, none) -> tensor<?x?x?x?xf32>
  // CHECK: onnx.Return [[RES]] : tensor<?x?x?x?xf32>
}

// -----

// COM: Test ResizeChannelLast with 3x upscale
func.func @test_xfe_resize_3x_upsample(%arg0: tensor<1x8x8x16xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 3.0, 3.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "asymmetric",
    mode = "nearest",
    nearest_mode = "ceil"
  } : (tensor<1x8x8x16xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_resize_3x_upsample
  // CHECK: [[RES:%.+]] = "onnx.XFEResize"(%arg0, %{{.*}}, %{{.*}}, %{{.*}}) {{{.*}}coordinate_transformation_mode = "asymmetric"{{.*}}mode = "nearest"{{.*}}nearest_mode = "ceil"{{.*}}} : (tensor<1x8x8x16xf32>, none, tensor<4xf32>, none) -> tensor<1x24x24x16xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x24x24x16xf32>
}

// -----

// COM: Test ResizeChannelLast linear interpolation mode
func.func @test_xfe_resize_linear(%arg0: tensor<1x4x4x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "align_corners",
    mode = "linear"
  } : (tensor<1x4x4x3xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: test_xfe_resize_linear
  // CHECK: [[RES:%.+]] = "onnx.XFEResize"(%arg0, %{{.*}}, %{{.*}}, %{{.*}}) {{{.*}}coordinate_transformation_mode = "align_corners"{{.*}}mode = "linear"{{.*}}} : (tensor<1x4x4x3xf32>, none, tensor<4xf32>, none) -> tensor<1x8x8x3xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x8x8x3xf32>
}

