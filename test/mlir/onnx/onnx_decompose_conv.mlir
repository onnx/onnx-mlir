// RUN: onnx-mlir-opt --decompose-onnx='enable-conv-to-matmul' --canonicalize %s -split-input-file | FileCheck %s

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


// CHECK-LABEL:  func.func @test_conv_2d_3x3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x32x32xf32>, [[PARAM_1_:%.+]]: tensor<64x3x3x3xf32>) -> tensor<1x64x30x30xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<64> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<-2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[64, 27]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Im2Col"([[PARAM_0_]]) <{auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x3x32x32xf32>) -> tensor<?x27x?xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Shape"([[VAR_9_]]) <{start = 0 : si64}> : (tensor<?x27x?xf32>) -> tensor<3xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Slice"([[VAR_10_]], [[VAR_6_]], [[VAR_5_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<3xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Concat"([[VAR_8_]], [[VAR_11_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_12_]]) <{allowzero = 0 : si64}> : (tensor<?x27x?xf32>, tensor<2xi64>) -> tensor<27x?xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_3_]]) <{allowzero = 0 : si64}> : (tensor<64x3x3x3xf32>, tensor<2xi64>) -> tensor<64x27xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.MatMul"([[VAR_14_]], [[VAR_13_]]) : (tensor<64x27xf32>, tensor<27x?xf32>) -> tensor<64x?xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_2_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Div"([[VAR_16_]], [[VAR_4_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_2_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Div"([[VAR_18_]], [[VAR_4_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Concat"([[VAR_4_]], [[VAR_1_]], [[VAR_17_]], [[VAR_19_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_21_:%.+]] = "onnx.Reshape"([[VAR_15_]], [[VAR_20_]]) <{allowzero = 0 : si64}> : (tensor<64x?xf32>, tensor<4xi64>) -> tensor<1x64x30x30xf32>
// CHECK:           return [[VAR_21_]] : tensor<1x64x30x30xf32>
// CHECK:         }
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


// CHECK-LABEL:  func.func @test_conv_with_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x32x32xf32>, [[PARAM_1_:%.+]]: tensor<64x3x3x3xf32>, [[PARAM_2_:%.+]]: tensor<64xf32>) -> tensor<1x64x30x30xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<64> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<-2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[64, 27]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Im2Col"([[PARAM_0_]]) <{auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x3x32x32xf32>) -> tensor<?x27x?xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Shape"([[VAR_9_]]) <{start = 0 : si64}> : (tensor<?x27x?xf32>) -> tensor<3xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Slice"([[VAR_10_]], [[VAR_6_]], [[VAR_5_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<3xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Concat"([[VAR_8_]], [[VAR_11_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_12_]]) <{allowzero = 0 : si64}> : (tensor<?x27x?xf32>, tensor<2xi64>) -> tensor<27x?xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_3_]]) <{allowzero = 0 : si64}> : (tensor<64x3x3x3xf32>, tensor<2xi64>) -> tensor<64x27xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Unsqueeze"([[PARAM_2_]], [[VAR_4_]]) : (tensor<64xf32>, tensor<1xi64>) -> tensor<64x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Gemm"([[VAR_14_]], [[VAR_13_]], [[VAR_15_]]) <{alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64}> : (tensor<64x27xf32>, tensor<27x?xf32>, tensor<64x1xf32>) -> tensor<64x?xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_2_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Div"([[VAR_17_]], [[VAR_4_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_2_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Div"([[VAR_19_]], [[VAR_4_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_21_:%.+]] = "onnx.Concat"([[VAR_4_]], [[VAR_1_]], [[VAR_18_]], [[VAR_20_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Reshape"([[VAR_16_]], [[VAR_21_]]) <{allowzero = 0 : si64}> : (tensor<64x?xf32>, tensor<4xi64>) -> tensor<1x64x30x30xf32>
// CHECK:           return [[VAR_22_]] : tensor<1x64x30x30xf32>
// CHECK:         }
// -----

// Test 3: 1x1 convolution should be decomposed to MatMul (not Im2Col)
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
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Im2Col"
// CHECK: "onnx.Reshape"
// CHECK: "onnx.MatMul"
// CHECK: "onnx.Reshape"

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
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: kernel_shape = [3, 3]
// CHECK-SAME: pads = [1, 1, 1, 1]
// CHECK-SAME: strides = [2, 2]
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


// CHECK-LABEL:  func.func @test_conv_5x5
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x32x32xf32>, [[PARAM_1_:%.+]]: tensor<64x3x5x5xf32>) -> tensor<1x64x28x28xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<64> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<-4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[64, 75]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Im2Col"([[PARAM_0_]]) <{auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [5, 5], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x3x32x32xf32>) -> tensor<?x75x?xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Shape"([[VAR_9_]]) <{start = 0 : si64}> : (tensor<?x75x?xf32>) -> tensor<3xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Slice"([[VAR_10_]], [[VAR_6_]], [[VAR_5_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<3xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Concat"([[VAR_8_]], [[VAR_11_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_12_]]) <{allowzero = 0 : si64}> : (tensor<?x75x?xf32>, tensor<2xi64>) -> tensor<75x?xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_3_]]) <{allowzero = 0 : si64}> : (tensor<64x3x5x5xf32>, tensor<2xi64>) -> tensor<64x75xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.MatMul"([[VAR_14_]], [[VAR_13_]]) : (tensor<64x75xf32>, tensor<75x?xf32>) -> tensor<64x?xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_2_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Div"([[VAR_16_]], [[VAR_4_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_2_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Div"([[VAR_18_]], [[VAR_4_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Concat"([[VAR_4_]], [[VAR_1_]], [[VAR_17_]], [[VAR_19_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_21_:%.+]] = "onnx.Reshape"([[VAR_15_]], [[VAR_20_]]) <{allowzero = 0 : si64}> : (tensor<64x?xf32>, tensor<4xi64>) -> tensor<1x64x28x28xf32>
// CHECK:           return [[VAR_21_]] : tensor<1x64x28x28xf32>
// CHECK:         }
// -----

// Test 8: 1x1 convolution with stride > 1 should decompose to Im2Col+MatMul
func.func @test_conv_1x1_stride(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<64x3x1x1xf32>) -> tensor<1x64x16x16xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    kernel_shape = [1, 1],
    strides = [2, 2],
    pads = [0, 0, 0, 0],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x32x32xf32>, tensor<64x3x1x1xf32>, none) -> tensor<1x64x16x16xf32>
  return %0 : tensor<1x64x16x16xf32>
}

// CHECK-LABEL: func.func @test_conv_1x1_stride
// CHECK: [[IM2COL:%.+]] = "onnx.Im2Col"(%arg0)
// CHECK-SAME: kernel_shape = [1, 1]
// CHECK-SAME: strides = [2, 2]
// CHECK: [[MATMUL:%.+]] = "onnx.MatMul"
// CHECK: [[RESHAPE:%.+]] = "onnx.Reshape"
// CHECK: return [[RESHAPE]]

// -----

// Test 9: 1x1 convolution with padding should decompose to Im2Col+MatMul
func.func @test_conv_1x1_padding(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<64x3x1x1xf32>) -> tensor<1x64x34x34xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    kernel_shape = [1, 1],
    strides = [1, 1],
    pads = [1, 1, 1, 1],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x32x32xf32>, tensor<64x3x1x1xf32>, none) -> tensor<1x64x34x34xf32>
  return %0 : tensor<1x64x34x34xf32>
}

// CHECK-LABEL: func.func @test_conv_1x1_padding
// CHECK: [[IM2COL:%.+]] = "onnx.Im2Col"(%arg0)
// CHECK-SAME: kernel_shape = [1, 1]
// CHECK-SAME: pads = [1, 1, 1, 1]
// CHECK: [[MATMUL:%.+]] = "onnx.MatMul"
// CHECK: [[RESHAPE:%.+]] = "onnx.Reshape"
// CHECK: return [[RESHAPE]]

// -----



// Test 10: 1x1 convolution with bias should decompose to MatMul with Add
func.func @test_conv_1x1_with_bias(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<64x3x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x32x32xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
    kernel_shape = [1, 1],
    strides = [1, 1],
    pads = [0, 0, 0, 0],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x32x32xf32>, tensor<64x3x1x1xf32>, tensor<64xf32>) -> tensor<1x64x32x32xf32>
  return %0 : tensor<1x64x32x32xf32>
}

// CHECK-LABEL:  func.func @test_conv_1x1_with_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x32x32xf32>, [[PARAM_1_:%.+]]: tensor<64x3x1x1xf32>, [[PARAM_2_:%.+]]: tensor<64xf32>) -> tensor<1x64x32x32xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<64> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[64, 3, 1, 1]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[1, 3, 1024]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1024> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<[1, 3, 32, 32]> : tensor<4xi64>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Slice"([[VAR_9_]], [[VAR_8_]], [[VAR_7_]], [[VAR_8_]], [[VAR_6_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Concat"([[VAR_10_]], [[VAR_5_]]) <{axis = 0 : si64}> : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_11_]]) <{allowzero = 0 : si64}> : (tensor<1x3x32x32xf32>, tensor<3xi64>) -> tensor<1x3x1024xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Slice"([[VAR_3_]], [[VAR_7_]], [[VAR_4_]], [[VAR_8_]], [[VAR_6_]]) : (tensor<3xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Concat"([[VAR_4_]], [[VAR_13_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Reshape"([[VAR_12_]], [[VAR_14_]]) <{allowzero = 0 : si64}> : (tensor<1x3x1024xf32>, tensor<2xi64>) -> tensor<3x1024xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_2_]], [[VAR_8_]], [[VAR_6_]], [[VAR_8_]], [[VAR_6_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Concat"([[VAR_16_]], [[VAR_4_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_17_]]) <{allowzero = 0 : si64}> : (tensor<64x3x1x1xf32>, tensor<2xi64>) -> tensor<64x3xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Unsqueeze"([[PARAM_2_]], [[VAR_6_]]) : (tensor<64xf32>, tensor<1xi64>) -> tensor<64x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Gemm"([[VAR_18_]], [[VAR_15_]], [[VAR_19_]]) <{alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64}> : (tensor<64x3xf32>, tensor<3x1024xf32>, tensor<64x1xf32>) -> tensor<64x1024xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Concat"([[VAR_6_]], [[VAR_1_]], [[VAR_0_]], [[VAR_0_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Reshape"([[VAR_20_]], [[VAR_21_]]) <{allowzero = 0 : si64}> : (tensor<64x1024xf32>, tensor<4xi64>) -> tensor<1x64x32x32xf32>
// CHECK:           return [[VAR_22_]] : tensor<1x64x32x32xf32>
// CHECK:         }
