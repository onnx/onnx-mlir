// RUN: onnx-mlir-opt --late-decompose %s -split-input-file | FileCheck %s

// -----

// Test 1: Basic 2D convolution with 3x3 kernel should be decomposed
func.func @test_conv_2d_3x3(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<64x3x3x3xf32>) -> tensor<1x64x30x30xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [0, 0, 0, 0],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x32x32xf32>, tensor<64x3x3x3xf32>, none) -> tensor<1x64x30x30xf32>
  return %0 : tensor<1x64x30x30xf32>
}

// CHECK-LABEL: func.func @test_conv_2d_3x3
// CHECK: [[IM2COL:%.+]] = "onnx.Im2Col"(%arg0)
// CHECK-SAME: kernel_shape = [3, 3]
// CHECK: [[RESHAPE_W:%.+]] = "onnx.Reshape"(%arg1
// CHECK: [[TRANSPOSE:%.+]] = "onnx.Transpose"([[RESHAPE_W]])
// CHECK: [[MATMUL:%.+]] = "onnx.MatMul"([[IM2COL]], [[TRANSPOSE]])
// CHECK: [[RESHAPE_OUT:%.+]] = "onnx.Reshape"([[MATMUL]]
// CHECK: return [[RESHAPE_OUT]]

// -----

// Test 2: Conv with bias should include Add operation
func.func @test_conv_with_bias(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x30x30xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [0, 0, 0, 0],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x32x32xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x30x30xf32>
  return %0 : tensor<1x64x30x30xf32>
}

// CHECK-LABEL: func.func @test_conv_with_bias
// CHECK: [[IM2COL:%.+]] = "onnx.Im2Col"
// CHECK: [[MATMUL:%.+]] = "onnx.MatMul"
// CHECK: [[ADD:%.+]] = "onnx.Add"([[MATMUL]], %arg2)
// CHECK: [[RESHAPE:%.+]] = "onnx.Reshape"([[ADD]]
// CHECK: return [[RESHAPE]]

// -----

// Test 3: 1x1 convolution should NOT be decomposed (handled by ConvOpt)
func.func @test_conv_1x1(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<64x3x1x1xf32>) -> tensor<1x64x32x32xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    kernel_shape = [1, 1],
    strides = [1, 1],
    pads = [0, 0, 0, 0],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x32x32xf32>, tensor<64x3x1x1xf32>, none) -> tensor<1x64x32x32xf32>
  return %0 : tensor<1x64x32x32xf32>
}

// CHECK-LABEL: func.func @test_conv_1x1
// CHECK: "onnx.Conv"
// CHECK-NOT: "onnx.Im2Col"

// -----

// Test 4: Grouped convolution should NOT be decomposed
func.func @test_grouped_conv(%arg0: tensor<1x6x32x32xf32>, %arg1: tensor<64x3x3x3xf32>) -> tensor<1x64x30x30xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [0, 0, 0, 0],
    dilations = [1, 1],
    group = 2 : si64
  } : (tensor<1x6x32x32xf32>, tensor<64x3x3x3xf32>, none) -> tensor<1x64x30x30xf32>
  return %0 : tensor<1x64x30x30xf32>
}

// CHECK-LABEL: func.func @test_grouped_conv
// CHECK: "onnx.Conv"
// CHECK-NOT: "onnx.Im2Col"

// -----

// Test 5: Conv with stride and padding
func.func @test_conv_stride_pad(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<64x3x3x3xf32>) -> tensor<1x64x16x16xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    strides = [2, 2],
    pads = [1, 1, 1, 1],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x32x32xf32>, tensor<64x3x3x3xf32>, none) -> tensor<1x64x16x16xf32>
  return %0 : tensor<1x64x16x16xf32>
}

// CHECK-LABEL: func.func @test_conv_stride_pad
// CHECK: [[IM2COL:%.+]] = "onnx.Im2Col"(%arg0)
// CHECK-SAME: kernel_shape = [3, 3]
// CHECK-SAME: strides = [2, 2]
// CHECK-SAME: pads = [1, 1, 1, 1]
// CHECK: [[MATMUL:%.+]] = "onnx.MatMul"
// CHECK: [[RESHAPE:%.+]] = "onnx.Reshape"
// CHECK: return [[RESHAPE]]

// -----

// Test 6: Conv with dilation
func.func @test_conv_dilation(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<64x3x3x3xf32>) -> tensor<1x64x28x28xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [0, 0, 0, 0],
    dilations = [2, 2],
    group = 1 : si64
  } : (tensor<1x3x32x32xf32>, tensor<64x3x3x3xf32>, none) -> tensor<1x64x28x28xf32>
  return %0 : tensor<1x64x28x28xf32>
}

// CHECK-LABEL: func.func @test_conv_dilation
// CHECK: [[IM2COL:%.+]] = "onnx.Im2Col"(%arg0)
// CHECK-SAME: dilations = [2, 2]
// CHECK: [[MATMUL:%.+]] = "onnx.MatMul"
// CHECK: [[RESHAPE:%.+]] = "onnx.Reshape"
// CHECK: return [[RESHAPE]]

// -----

// Test 7: 5x5 kernel convolution
func.func @test_conv_5x5(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<64x3x5x5xf32>) -> tensor<1x64x28x28xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    kernel_shape = [5, 5],
    strides = [1, 1],
    pads = [0, 0, 0, 0],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x32x32xf32>, tensor<64x3x5x5xf32>, none) -> tensor<1x64x28x28xf32>
  return %0 : tensor<1x64x28x28xf32>
}

// CHECK-LABEL: func.func @test_conv_5x5
// CHECK: [[IM2COL:%.+]] = "onnx.Im2Col"(%arg0)
// CHECK-SAME: kernel_shape = [5, 5]
// CHECK: [[RESHAPE_W:%.+]] = "onnx.Reshape"(%arg1
// CHECK: [[TRANSPOSE:%.+]] = "onnx.Transpose"([[RESHAPE_W]])
// CHECK: [[MATMUL:%.+]] = "onnx.MatMul"([[IM2COL]], [[TRANSPOSE]])
// CHECK: [[RESHAPE_OUT:%.+]] = "onnx.Reshape"([[MATMUL]]
// CHECK: return [[RESHAPE_OUT]]