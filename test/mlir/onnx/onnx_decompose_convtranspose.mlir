// RUN: onnx-mlir-opt --shape-inference --decompose-onnx %s -split-input-file | FileCheck %s

// REQUIRES: decomp_onnx_convtranspose

// -----

// Test unit strides. Only convert weight tensor

  func.func @test_convtrans_unitstrides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x5x5xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x5x5xf32>
    onnx.Return %1 : tensor<1x2x5x5xf32>

// CHECK-LABEL:  func.func @test_convtrans_unitstrides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x5x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x2x3x3xf32>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReverseSequence"([[VAR_3_]], [[VAR_1_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReverseSequence"([[VAR_4_]], [[VAR_1_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x2xf32>) -> tensor<1x2x3x3xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0, 2, 3]} : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_7_]], [[VAR_2_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 2, 2, 2]} : (tensor<1x1x3x3xf32>, tensor<2x1x3x3xf32>, none) -> tensor<1x2x5x5xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Pad"([[VAR_8_]], [[VAR_0_]], [[VAR_2_]], [[VAR_2_]]) {mode = "constant"} : (tensor<1x2x5x5xf32>, tensor<8xi64>, none, none) -> tensor<1x2x5x5xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Pad"([[VAR_9_]], [[VAR_0_]], [[VAR_2_]], [[VAR_2_]]) {mode = "constant"} : (tensor<1x2x5x5xf32>, tensor<8xi64>, none, none) -> tensor<1x2x5x5xf32>
// CHECK:           onnx.Return [[VAR_10_]] : tensor<1x2x5x5xf32>

  }

// -----

// Test 1d input

  func.func @test_convtrans1d_unitstrides(%arg0: tensor<1x1x3xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x5xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3xf32>, tensor<1x2x3xf32>, none) -> tensor<1x2x5xf32>
    onnx.Return %1 : tensor<1x2x5xf32>

// CHECK-LABEL:  func.func @test_convtrans1d_unitstrides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3xf32>) -> tensor<1x2x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<6xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<1x2x3xf32>) -> tensor<3x1x2xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReverseSequence"([[VAR_3_]], [[VAR_1_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x1x2xf32>, tensor<1xi64>) -> tensor<3x1x2xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 2, 0]} : (tensor<3x1x2xf32>) -> tensor<1x2x3xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0, 2]} : (tensor<1x2x3xf32>) -> tensor<2x1x3xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_6_]], [[VAR_2_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 2]} : (tensor<1x1x3xf32>, tensor<2x1x3xf32>, none) -> tensor<1x2x5xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Pad"([[VAR_7_]], [[VAR_0_]], [[VAR_2_]], [[VAR_2_]]) {mode = "constant"} : (tensor<1x2x5xf32>, tensor<6xi64>, none, none) -> tensor<1x2x5xf32>
// CHECK:           onnx.Return [[VAR_8_]] : tensor<1x2x5xf32>
  }

// -----

// Test 3d input

  func.func @test_convtrans3d_unitstrides(%arg0: tensor<1x1x3x4x5xf32>, %arg1: tensor<1x2x3x3x3xf32>) -> tensor<1x2x5x6x7xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3x4x5xf32>, tensor<1x2x3x3x3xf32>, none) -> tensor<1x2x5x6x7xf32>
    onnx.Return %1 : tensor<1x2x5x6x7xf32>

// CHECK-LABEL:  func.func @test_convtrans3d_unitstrides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x4x5xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3x3xf32>) -> tensor<1x2x5x6x7xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<10xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 4, 0, 1]} : (tensor<1x2x3x3x3xf32>) -> tensor<3x3x3x1x2xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReverseSequence"([[VAR_4_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x3x1x2xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.ReverseSequence"([[VAR_5_]], [[VAR_2_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x3x1x2xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [2, 3, 4, 0, 1]} : (tensor<3x3x3x1x2xf32>) -> tensor<3x1x2x3x3xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.ReverseSequence"([[VAR_7_]], [[VAR_1_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x1x2x3x3xf32>, tensor<1xi64>) -> tensor<3x1x2x3x3xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 2, 3, 4, 0]} : (tensor<3x1x2x3x3xf32>) -> tensor<1x2x3x3x3xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0, 2, 3, 4]} : (tensor<1x2x3x3x3xf32>) -> tensor<2x1x3x3x3xf32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_10_]], [[VAR_3_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 2, 2, 2, 2, 2]} : (tensor<1x1x3x4x5xf32>, tensor<2x1x3x3x3xf32>, none) -> tensor<1x2x5x6x7xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Pad"([[VAR_11_]], [[VAR_0_]], [[VAR_3_]], [[VAR_3_]]) {mode = "constant"} : (tensor<1x2x5x6x7xf32>, tensor<10xi64>, none, none) -> tensor<1x2x5x6x7xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Pad"([[VAR_12_]], [[VAR_0_]], [[VAR_3_]], [[VAR_3_]]) {mode = "constant"} : (tensor<1x2x5x6x7xf32>, tensor<10xi64>, none, none) -> tensor<1x2x5x6x7xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Pad"([[VAR_13_]], [[VAR_0_]], [[VAR_3_]], [[VAR_3_]]) {mode = "constant"} : (tensor<1x2x5x6x7xf32>, tensor<10xi64>, none, none) -> tensor<1x2x5x6x7xf32>
// CHECK:           onnx.Return [[VAR_14_]] : tensor<1x2x5x6x7xf32>
  }

// -----

// Test non unit strides. Added pads between elements  in input data.

  func.func @test_convtrans_strides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 2, 1, 2], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x7x3xf32>
    onnx.Return %1 : tensor<1x2x7x3xf32>

// CHECK-LABEL:  func.func @test_convtrans_strides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x2x3x3xf32>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.ReverseSequence"([[VAR_6_]], [[VAR_4_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.ReverseSequence"([[VAR_7_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x2xf32>) -> tensor<1x2x3x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0, 2, 3]} : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK-DAG:       [[VAR_11_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_3_]]) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<3xi64>) -> (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Pad"([[VAR_11_]]#0, [[VAR_2_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, none, none) -> tensor<1x1x3x3xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Pad"([[VAR_11_]]#1, [[VAR_2_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, none, none) -> tensor<1x1x3x3xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Concat"([[VAR_12_]], [[VAR_13_]], [[VAR_11_]]#2) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x7x3xf32>
// CHECK:           [[VAR_15_:%.+]]:3 = "onnx.Split"([[VAR_14_]], [[VAR_3_]]) {axis = 3 : si64} : (tensor<1x1x7x3xf32>, tensor<3xi64>) -> (tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>)
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Pad"([[VAR_15_]]#0, [[VAR_1_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, none, none) -> tensor<1x1x7x2xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Pad"([[VAR_15_]]#1, [[VAR_1_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, none, none) -> tensor<1x1x7x2xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_16_]], [[VAR_17_]], [[VAR_15_]]#2) {axis = 3 : si64} : (tensor<1x1x7x2xf32>, tensor<1x1x7x2xf32>, tensor<1x1x7x1xf32>) -> tensor<1x1x7x5xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Conv"([[VAR_18_]], [[VAR_10_]], [[VAR_5_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 0, 1, 0]} : (tensor<1x1x7x5xf32>, tensor<2x1x3x3xf32>, none) -> tensor<1x2x7x3xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Pad"([[VAR_19_]], [[VAR_0_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x2x7x3xf32>, tensor<8xi64>, none, none) -> tensor<1x2x7x3xf32>
// CHECK:           [[VAR_21_:%.+]] = "onnx.Pad"([[VAR_20_]], [[VAR_0_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x2x7x3xf32>, tensor<8xi64>, none, none) -> tensor<1x2x7x3xf32>
// CHECK:           onnx.Return [[VAR_21_]] : tensor<1x2x7x3xf32>
  }

// -----

// Test output_padding. Additional pads are inserted after Conv op

  func.func @test_convtrans_outputpadding(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x10x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, output_shape = [10, 8], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x10x8xf32>
    onnx.Return %1 : tensor<1x2x10x8xf32>

// CHECK-LABEL:  func.func @test_convtrans_outputpadding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x10x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 1, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x2x3x3xf32>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.ReverseSequence"([[VAR_6_]], [[VAR_4_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.ReverseSequence"([[VAR_7_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x2xf32>) -> tensor<1x2x3x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0, 2, 3]} : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK-DAG:       [[VAR_11_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_3_]]) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<3xi64>) -> (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Pad"([[VAR_11_]]#0, [[VAR_2_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, none, none) -> tensor<1x1x3x3xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Pad"([[VAR_11_]]#1, [[VAR_2_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, none, none) -> tensor<1x1x3x3xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Concat"([[VAR_12_]], [[VAR_13_]], [[VAR_11_]]#2) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x7x3xf32>
// CHECK:           [[VAR_15_:%.+]]:3 = "onnx.Split"([[VAR_14_]], [[VAR_3_]]) {axis = 3 : si64} : (tensor<1x1x7x3xf32>, tensor<3xi64>) -> (tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>)
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Pad"([[VAR_15_]]#0, [[VAR_1_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, none, none) -> tensor<1x1x7x2xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Pad"([[VAR_15_]]#1, [[VAR_1_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, none, none) -> tensor<1x1x7x2xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_16_]], [[VAR_17_]], [[VAR_15_]]#2) {axis = 3 : si64} : (tensor<1x1x7x2xf32>, tensor<1x1x7x2xf32>, tensor<1x1x7x1xf32>) -> tensor<1x1x7x5xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Conv"([[VAR_18_]], [[VAR_10_]], [[VAR_5_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 2, 2, 2]} : (tensor<1x1x7x5xf32>, tensor<2x1x3x3xf32>, none) -> tensor<1x2x9x7xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Pad"([[VAR_19_]], [[VAR_0_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x2x9x7xf32>, tensor<8xi64>, none, none) -> tensor<1x2x10x7xf32>
// CHECK:           [[VAR_21_:%.+]] = "onnx.Pad"([[VAR_20_]], [[VAR_1_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<1x2x10x7xf32>, tensor<8xi64>, none, none) -> tensor<1x2x10x8xf32>
// CHECK:           onnx.Return [[VAR_21_]] : tensor<1x2x10x8xf32>
  }

// -----

// Test for unknown dimension in spatial dimensions

  func.func @test_convtranspose_unknown_spatial_dim(%arg0: tensor<?x?x3x3xf32>, %arg1: tensor<?x?x3x3xf32>) -> tensor<?x?x10x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "test", output_padding = [1, 1], output_shape = [10, 8], strides = [3, 2]} : (tensor<?x?x3x3xf32>, tensor<?x?x3x3xf32>, none) -> tensor<?x?x10x8xf32>
    onnx.Return %1 : tensor<?x?x10x8xf32>

// CHECK-LABEL:  func.func @test_convtranspose_unknown_spatial_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3x3xf32>, [[PARAM_1_:%.+]]: tensor<?x?x3x3xf32>) -> tensor<?x?x10x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 1, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<?x?x3x3xf32>) -> tensor<3x3x?x?xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.ReverseSequence"([[VAR_6_]], [[VAR_4_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x?x?xf32>, tensor<3xi64>) -> tensor<3x3x?x?xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.ReverseSequence"([[VAR_7_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x?x?xf32>, tensor<3xi64>) -> tensor<3x3x?x?xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x?x?xf32>) -> tensor<?x?x3x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0, 2, 3]} : (tensor<?x?x3x3xf32>) -> tensor<?x?x3x3xf32>
// CHECK-DAG:       [[VAR_11_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_3_]]) {axis = 2 : si64} : (tensor<?x?x3x3xf32>, tensor<3xi64>) -> (tensor<?x?x1x3xf32>, tensor<?x?x1x3xf32>, tensor<?x?x1x3xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Pad"([[VAR_11_]]#0, [[VAR_2_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<?x?x1x3xf32>, tensor<8xi64>, none, none) -> tensor<?x?x3x3xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Pad"([[VAR_11_]]#1, [[VAR_2_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<?x?x1x3xf32>, tensor<8xi64>, none, none) -> tensor<?x?x3x3xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Concat"([[VAR_12_]], [[VAR_13_]], [[VAR_11_]]#2) {axis = 2 : si64} : (tensor<?x?x3x3xf32>, tensor<?x?x3x3xf32>, tensor<?x?x1x3xf32>) -> tensor<?x?x7x3xf32>
// CHECK:           [[VAR_15_:%.+]]:3 = "onnx.Split"([[VAR_14_]], [[VAR_3_]]) {axis = 3 : si64} : (tensor<?x?x7x3xf32>, tensor<3xi64>) -> (tensor<?x?x7x1xf32>, tensor<?x?x7x1xf32>, tensor<?x?x7x1xf32>)
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Pad"([[VAR_15_]]#0, [[VAR_1_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<?x?x7x1xf32>, tensor<8xi64>, none, none) -> tensor<?x?x7x2xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Pad"([[VAR_15_]]#1, [[VAR_1_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<?x?x7x1xf32>, tensor<8xi64>, none, none) -> tensor<?x?x7x2xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_16_]], [[VAR_17_]], [[VAR_15_]]#2) {axis = 3 : si64} : (tensor<?x?x7x2xf32>, tensor<?x?x7x2xf32>, tensor<?x?x7x1xf32>) -> tensor<?x?x7x5xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Conv"([[VAR_18_]], [[VAR_10_]], [[VAR_5_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [2, 2, 2, 2]} : (tensor<?x?x7x5xf32>, tensor<?x?x3x3xf32>, none) -> tensor<?x?x9x7xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Pad"([[VAR_19_]], [[VAR_0_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<?x?x9x7xf32>, tensor<8xi64>, none, none) -> tensor<?x?x10x7xf32>
// CHECK:           [[VAR_21_:%.+]] = "onnx.Pad"([[VAR_20_]], [[VAR_1_]], [[VAR_5_]], [[VAR_5_]]) {mode = "constant"} : (tensor<?x?x10x7xf32>, tensor<8xi64>, none, none) -> tensor<?x?x10x8xf32>
// CHECK:           onnx.Return [[VAR_21_]] : tensor<?x?x10x8xf32>
  }
