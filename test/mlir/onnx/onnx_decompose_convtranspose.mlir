// RUN: onnx-mlir-opt --shape-inference --decompose-onnx %s -split-input-file | FileCheck %s

// -----

// Test unit strides. Only convert weight tensor

  func.func @test_convtrans_unitstrides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x5x5xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x5x5xf32>
    onnx.Return %1 : tensor<1x2x5x5xf32>
  }

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_convtrans_unitstrides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x5x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.UpsampleAndPad"([[PARAM_0_]]) <{pads = [2, 2, 2, 2], strides = [1, 1]}> : (tensor<1x1x3x3xf32>) -> tensor<*xf32>
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
// CHECK:           [[VAR_12_:%.+]] = "onnx.Conv"([[VAR_1_]], [[VAR_11_]], [[VAR_0_]]) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<*xf32>, tensor<2x1x3x3xf32>, none) -> tensor<1x2x5x5xf32>
// CHECK:           onnx.Return [[VAR_12_]] : tensor<1x2x5x5xf32>
// CHECK:         }

// -----

// Test 1d input

  func.func @test_convtrans1d_unitstrides(%arg0: tensor<1x1x3xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x5xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3xf32>, tensor<1x2x3xf32>, none) -> tensor<1x2x5xf32>
    onnx.Return %1 : tensor<1x2x5xf32>
  }

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_convtrans1d_unitstrides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3xf32>) -> tensor<1x2x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.UpsampleAndPad"([[PARAM_0_]]) <{pads = [2, 2], strides = [1]}> : (tensor<1x1x3xf32>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 1, 2, 3]> : tensor<4xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<1x2x3xf32>, tensor<4xi64>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<-9223372036854775808> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Slice"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<1x1x2x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) <{perm = [0, 2, 1, 3]}> : (tensor<1x1x2x3xf32>) -> tensor<1x2x1x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[2, 1, 3]> : tensor<3xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_10_]]) <{allowzero = 0 : si64}> : (tensor<1x2x1x3xf32>, tensor<3xi64>) -> tensor<2x1x3xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Conv"([[VAR_1_]], [[VAR_1_]]1, [[VAR_0_]]) <{auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [3], pads = [0, 0], strides = [1]}> : (tensor<*xf32>, tensor<2x1x3xf32>, none) -> tensor<1x2x5xf32>
// CHECK:           onnx.Return [[VAR_12_]] : tensor<1x2x5xf32>
// CHECK:         }

// -----

// Test 3d input

  func.func @test_convtrans3d_unitstrides(%arg0: tensor<1x1x3x4x5xf32>, %arg1: tensor<1x2x3x3x3xf32>) -> tensor<1x2x5x6x7xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3x4x5xf32>, tensor<1x2x3x3x3xf32>, none) -> tensor<1x2x5x6x7xf32>
    onnx.Return %1 : tensor<1x2x5x6x7xf32>
  }

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_convtrans3d_unitstrides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x4x5xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3x3xf32>) -> tensor<1x2x5x6x7xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.UpsampleAndPad"([[PARAM_0_]]) <{pads = [2, 2, 2, 2, 2, 2], strides = [1, 1, 1]}> : (tensor<1x1x3x4x5xf32>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 1, 2, 3, 3, 3]> : tensor<6xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<1x2x3x3x3xf32>, tensor<6xi64>) -> tensor<1x1x2x3x3x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<3xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<-9223372036854775808> : tensor<3xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[3, 4, 5]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<-1> : tensor<3xi64>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Slice"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<1x1x2x3x3x3xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x1x2x3x3x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) <{perm = [0, 2, 1, 3, 4, 5]}> : (tensor<1x1x2x3x3x3xf32>) -> tensor<1x2x1x3x3x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[2, 1, 3, 3, 3]> : tensor<5xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_10_]]) <{allowzero = 0 : si64}> : (tensor<1x2x1x3x3x3xf32>, tensor<5xi64>) -> tensor<2x1x3x3x3xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Conv"([[VAR_1_]], [[VAR_1_]]1, [[VAR_0_]]) <{auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [3, 3, 3], pads = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1]}> : (tensor<*xf32>, tensor<2x1x3x3x3xf32>, none) -> tensor<1x2x5x6x7xf32>
// CHECK:           onnx.Return [[VAR_12_]] : tensor<1x2x5x6x7xf32>
// CHECK:         }

// -----

// Test non unit strides. Added pads between elements  in input data.

  func.func @test_convtrans_strides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 2, 1, 2], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x7x3xf32>
    onnx.Return %1 : tensor<1x2x7x3xf32>
  }

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_convtrans_strides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.UpsampleAndPad"([[PARAM_0_]]) <{pads = [1, 0, 1, 0], strides = [3, 2]}> : (tensor<1x1x3x3xf32>) -> tensor<*xf32>
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
// CHECK:           [[VAR_12_:%.+]] = "onnx.Conv"([[VAR_1_]], [[VAR_1_]]1, [[VAR_0_]]) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<*xf32>, tensor<2x1x3x3xf32>, none) -> tensor<1x2x7x3xf32>
// CHECK:           onnx.Return [[VAR_12_]] : tensor<1x2x7x3xf32>
// CHECK:         }

// -----

// Test output_padding. Additional pads are inserted after Conv op

  func.func @test_convtrans_outputpadding(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x10x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, output_shape = [10, 8], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x10x8xf32>
    onnx.Return %1 : tensor<1x2x10x8xf32>
  }

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_convtrans_outputpadding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x10x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.UpsampleAndPad"([[PARAM_0_]]) <{pads = [2, 2, 2, 2], strides = [3, 2]}> : (tensor<1x1x3x3xf32>) -> tensor<*xf32>
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
// CHECK:           [[VAR_12_:%.+]] = "onnx.Conv"([[VAR_1_]], [[VAR_1_]]1, [[VAR_0_]]) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<*xf32>, tensor<2x1x3x3xf32>, none) -> tensor<1x2x10x8xf32>
// CHECK:           onnx.Return [[VAR_12_]] : tensor<1x2x10x8xf32>
// CHECK:         }
