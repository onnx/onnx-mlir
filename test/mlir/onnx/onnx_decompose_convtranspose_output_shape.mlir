// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

// -----

// Test ConvTranspose decomposition with output_shape parameter.
// This test verifies that when output_shape is provided, the pads are correctly
// calculated to achieve the desired output dimensions.

func.func @test_convtranspose_output_shape(%arg0: tensor<1x1x5x5xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<1x1x12x12xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %none) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], output_shape = [12, 12], strides = [2, 2]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x12x12xf32>
  return %0 : tensor<1x1x12x12xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_convtranspose_output_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x5x5xf32>, [[PARAM_1_:%.+]]: tensor<1x1x3x3xf32>) -> tensor<1x1x12x12xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.UpsampleAndPad"([[PARAM_0_]]) <{pads = [2, 2, 3, 3], strides = [2, 2]}> : (tensor<1x1x5x5xf32>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 1, 1, 3, 3]> : tensor<5xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<1x1x3x3xf32>, tensor<5xi64>) -> tensor<1x1x1x3x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<-9223372036854775808> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[3, 4]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<-1> : tensor<2xi64>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Slice"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<1x1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x3x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) <{perm = [0, 2, 1, 3, 4]}> : (tensor<1x1x1x3x3xf32>) -> tensor<1x1x1x3x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[1, 1, 3, 3]> : tensor<4xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_10_]]) <{allowzero = 0 : si64}> : (tensor<1x1x1x3x3xf32>, tensor<4xi64>) -> tensor<1x1x3x3xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Conv"([[VAR_1_]], [[VAR_1_]]1, [[VAR_0_]]) <{auto_pad = "VALID", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<*xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x12x12xf32>
// CHECK:           return [[VAR_12_]] : tensor<1x1x12x12xf32>
// CHECK:         }
}

// -----

// Test ConvTranspose with output_shape and SAME_UPPER auto_pad.

func.func @test_convtranspose_output_shape_same_upper(%arg0: tensor<1x2x4x4xf32>, %arg1: tensor<2x3x3x3xf32>) -> tensor<1x3x10x10xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %none) {auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], output_shape = [10, 10], strides = [2, 2]} : (tensor<1x2x4x4xf32>, tensor<2x3x3x3xf32>, none) -> tensor<1x3x10x10xf32>
  return %0 : tensor<1x3x10x10xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_convtranspose_output_shape_same_upper
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x4x4xf32>, [[PARAM_1_:%.+]]: tensor<2x3x3x3xf32>) -> tensor<1x3x10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.UpsampleAndPad"([[PARAM_0_]]) <{pads = [2, 2, 3, 3], strides = [2, 2]}> : (tensor<1x2x4x4xf32>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 2, 3, 3, 3]> : tensor<5xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<2x3x3x3xf32>, tensor<5xi64>) -> tensor<1x2x3x3x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<-9223372036854775808> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[3, 4]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<-1> : tensor<2xi64>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Slice"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<1x2x3x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2x3x3x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) <{perm = [0, 2, 1, 3, 4]}> : (tensor<1x2x3x3x3xf32>) -> tensor<1x3x2x3x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[3, 2, 3, 3]> : tensor<4xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_10_]]) <{allowzero = 0 : si64}> : (tensor<1x3x2x3x3xf32>, tensor<4xi64>) -> tensor<3x2x3x3xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Conv"([[VAR_1_]], [[VAR_1_]]1, [[VAR_0_]]) <{auto_pad = "VALID", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<*xf32>, tensor<3x2x3x3xf32>, none) -> tensor<1x3x10x10xf32>
// CHECK:           return [[VAR_12_]] : tensor<1x3x10x10xf32>
// CHECK:         }
}

// -----

// Test ConvTranspose with output_shape requiring positive pads.

func.func @test_convtranspose_output_shape_positive_pads(%arg0: tensor<1x1x5x5xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<1x1x10x10xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %none) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], output_shape = [10, 10], strides = [2, 2]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x10x10xf32>
  return %0 : tensor<1x1x10x10xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_convtranspose_output_shape_positive_pads
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x5x5xf32>, [[PARAM_1_:%.+]]: tensor<1x1x3x3xf32>) -> tensor<1x1x10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.UpsampleAndPad"([[PARAM_0_]]) <{pads = [2, 2, 1, 1], strides = [2, 2]}> : (tensor<1x1x5x5xf32>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 1, 1, 3, 3]> : tensor<5xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<1x1x3x3xf32>, tensor<5xi64>) -> tensor<1x1x1x3x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<-9223372036854775808> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[3, 4]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<-1> : tensor<2xi64>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Slice"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<1x1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x3x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) <{perm = [0, 2, 1, 3, 4]}> : (tensor<1x1x1x3x3xf32>) -> tensor<1x1x1x3x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[1, 1, 3, 3]> : tensor<4xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_10_]]) <{allowzero = 0 : si64}> : (tensor<1x1x1x3x3xf32>, tensor<4xi64>) -> tensor<1x1x3x3xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Conv"([[VAR_1_]], [[VAR_1_]]1, [[VAR_0_]]) <{auto_pad = "VALID", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<*xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x10x10xf32>
// CHECK:           return [[VAR_12_]] : tensor<1x1x10x10xf32>
// CHECK:         }
}
// -----

// Test ConvTranspose with non-uniform strides and output_shape.
// This test case matches the user's MLIR example:
// Input: 1x1x3x3, Weight: 1x2x3x3, strides=(3,2), output_shape=(10,8)

func.func @test_convtranspose_nonuniform_strides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x10x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %none) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], output_shape = [10, 8], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x10x8xf32>
  return %0 : tensor<1x2x10x8xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_convtranspose_nonuniform_strides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x10x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.UpsampleAndPad"([[PARAM_0_]]) <{pads = [2, 2, 3, 3], strides = [3, 2]}> : (tensor<1x1x3x3xf32>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 1, 2, 3, 3]> : tensor<5xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<1x2x3x3xf32>, tensor<5xi64>) -> tensor<1x1x2x3x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<-9223372036854775808> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[3, 4]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<-1> : tensor<2xi64>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Slice"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<1x1x2x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x2x3x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) <{perm = [0, 2, 1, 3, 4]}> : (tensor<1x1x2x3x3xf32>) -> tensor<1x2x1x3x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[2, 1, 3, 3]> : tensor<4xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_10_]]) <{allowzero = 0 : si64}> : (tensor<1x2x1x3x3xf32>, tensor<4xi64>) -> tensor<2x1x3x3xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Conv"([[VAR_1_]], [[VAR_1_]]1, [[VAR_0_]]) <{auto_pad = "VALID", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<*xf32>, tensor<2x1x3x3xf32>, none) -> tensor<1x2x10x8xf32>
// CHECK:           return [[VAR_12_]] : tensor<1x2x10x8xf32>
// CHECK:         }
}

