// RUN: onnx-mlir-opt --shape-inference --decompose-onnx %s -split-input-file | FileCheck %s

// -----

// Test unit strides. Only convert weight tensor

  func.func @test_convtrans_unitstrides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x5x5xf32> {
// CHECK-LABEL:  func.func @test_convtrans_unitstrides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>)

    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x5x5xf32>
    onnx.Return %1 : tensor<1x2x5x5xf32>

// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x2x3x3xf32>) -> tensor<3x3x1x2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.ReverseSequence"([[VAR_1_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReverseSequence"([[VAR_3_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x2xf32>) -> tensor<1x2x3x3xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0, 2, 3]} : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_7_]], [[VAR_0_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 2, 2, 2]} : (tensor<1x1x3x3xf32>, tensor<2x1x3x3xf32>, none) -> tensor<1x2x5x5xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<0> : tensor<8xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Pad"([[VAR_8_]], [[VAR_9_]], [[VAR_10_]], [[VAR_11_]]) {mode = "constant"} : (tensor<1x2x5x5xf32>, tensor<8xi64>, none, none) -> tensor<1x2x5x5xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<0> : tensor<8xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_16_:%.+]] = "onnx.Pad"([[VAR_12_]], [[VAR_13_]], [[VAR_14_]], [[VAR_15_]]) {mode = "constant"} : (tensor<1x2x5x5xf32>, tensor<8xi64>, none, none) -> tensor<1x2x5x5xf32>
// CHECK:           onnx.Return [[VAR_16_]] : tensor<1x2x5x5xf32>

  }

// -----

// Test 1d input

  func.func @test_convtrans1d_unitstrides(%arg0: tensor<1x1x3xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x5xf32> {
// CHECK-LABEL:  func.func @test_convtrans1d_unitstrides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3xf32>)

    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3xf32>, tensor<1x2x3xf32>, none) -> tensor<1x2x5xf32>
    onnx.Return %1 : tensor<1x2x5xf32>

// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<1x2x3xf32>) -> tensor<3x1x2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReverseSequence"([[VAR_1_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x1x2xf32>, tensor<1xi64>) -> tensor<3x1x2xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 2, 0]} : (tensor<3x1x2xf32>) -> tensor<1x2x3xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0, 2]} : (tensor<1x2x3xf32>) -> tensor<2x1x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_5_]], [[VAR_0_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 2]} : (tensor<1x1x3xf32>, tensor<2x1x3xf32>, none) -> tensor<1x2x5xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<6xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_10_:%.+]] = "onnx.Pad"([[VAR_6_]], [[VAR_7_]], [[VAR_8_]], [[VAR_9_]]) {mode = "constant"} : (tensor<1x2x5xf32>, tensor<6xi64>, none, none) -> tensor<1x2x5xf32>
// CHECK:           onnx.Return [[VAR_10_]] : tensor<1x2x5xf32>
  }

// -----

// Test 3d input

  func.func @test_convtrans3d_unitstrides(%arg0: tensor<1x1x3x4x5xf32>, %arg1: tensor<1x2x3x3x3xf32>) -> tensor<1x2x5x6x7xf32> {
// CHECK-LABEL:  func.func @test_convtrans3d_unitstrides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x4x5xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3x3xf32>)

    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3x4x5xf32>, tensor<1x2x3x3x3xf32>, none) -> tensor<1x2x5x6x7xf32>
    onnx.Return %1 : tensor<1x2x5x6x7xf32>

// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 4, 0, 1]} : (tensor<1x2x3x3x3xf32>) -> tensor<3x3x3x1x2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.ReverseSequence"([[VAR_1_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x3x1x2xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReverseSequence"([[VAR_3_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x3x1x2xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [2, 3, 4, 0, 1]} : (tensor<3x3x3x1x2xf32>) -> tensor<3x1x2x3x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           [[VAR_8_:%.+]] = "onnx.ReverseSequence"([[VAR_6_]], [[VAR_7_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x1x2x3x3xf32>, tensor<1xi64>) -> tensor<3x1x2x3x3xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 2, 3, 4, 0]} : (tensor<3x1x2x3x3xf32>) -> tensor<1x2x3x3x3xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0, 2, 3, 4]} : (tensor<1x2x3x3x3xf32>) -> tensor<2x1x3x3x3xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_10_]], [[VAR_0_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 2, 2, 2, 2, 2]} : (tensor<1x1x3x4x5xf32>, tensor<2x1x3x3x3xf32>, none) -> tensor<1x2x5x6x7xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<0> : tensor<10xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Pad"([[VAR_11_]], [[VAR_12_]], [[VAR_13_]], [[VAR_14_]]) {mode = "constant"} : (tensor<1x2x5x6x7xf32>, tensor<10xi64>, none, none) -> tensor<1x2x5x6x7xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<0> : tensor<10xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Pad"([[VAR_15_]], [[VAR_16_]], [[VAR_17_]], [[VAR_18_]]) {mode = "constant"} : (tensor<1x2x5x6x7xf32>, tensor<10xi64>, none, none) -> tensor<1x2x5x6x7xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<0> : tensor<10xi64>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_23_:%.+]] = "onnx.Pad"([[VAR_19_]], [[VAR_20_]], [[VAR_21_]], [[VAR_22_]]) {mode = "constant"} : (tensor<1x2x5x6x7xf32>, tensor<10xi64>, none, none) -> tensor<1x2x5x6x7xf32>
// CHECK:           onnx.Return [[VAR_23_]] : tensor<1x2x5x6x7xf32>
  }

// -----

// Test non unit strides. Added pads between elements  in input data.

  func.func @test_convtrans_strides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> {
// CHECK-LABEL:  func.func @test_convtrans_strides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>)

    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 2, 1, 2], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x7x3xf32>
    onnx.Return %1 : tensor<1x2x7x3xf32>

// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x2x3x3xf32>) -> tensor<3x3x1x2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.ReverseSequence"([[VAR_1_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReverseSequence"([[VAR_3_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x2xf32>) -> tensor<1x2x3x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0, 2, 3]} : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_8_]]) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<3xi64>) -> (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>)
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Pad"([[VAR_9_]]#0, [[VAR_10_]], [[VAR_11_]], [[VAR_12_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, none, none) -> tensor<1x1x3x3xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_17_:%.+]] = "onnx.Pad"([[VAR_9_]]#1, [[VAR_14_]], [[VAR_15_]], [[VAR_16_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, none, none) -> tensor<1x1x3x3xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_13_]], [[VAR_17_]], [[VAR_9_]]#2) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x7x3xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]]:3 = "onnx.Split"([[VAR_18_]], [[VAR_19_]]) {axis = 3 : si64} : (tensor<1x1x7x3xf32>, tensor<3xi64>) -> (tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>)
// CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Pad"([[VAR_20_]]#0, [[VAR_21_]], [[VAR_22_]], [[VAR_23_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, none, none) -> tensor<1x1x7x2xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_28_:%.+]] = "onnx.Pad"([[VAR_20_]]#1, [[VAR_25_]], [[VAR_26_]], [[VAR_27_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, none, none) -> tensor<1x1x7x2xf32>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Concat"([[VAR_24_]], [[VAR_28_]], [[VAR_20_]]#2) {axis = 3 : si64} : (tensor<1x1x7x2xf32>, tensor<1x1x7x2xf32>, tensor<1x1x7x1xf32>) -> tensor<1x1x7x5xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Conv"([[VAR_29_]], [[VAR_7_]], [[VAR_0_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 0, 1, 0]} : (tensor<1x1x7x5xf32>, tensor<2x1x3x3xf32>, none) -> tensor<1x2x7x3xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = onnx.Constant dense<0> : tensor<8xi64>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Pad"([[VAR_30_]], [[VAR_31_]], [[VAR_32_]], [[VAR_33_]]) {mode = "constant"} : (tensor<1x2x7x3xf32>, tensor<8xi64>, none, none) -> tensor<1x2x7x3xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = onnx.Constant dense<0> : tensor<8xi64>
// CHECK-DAG:       [[VAR_36_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_38_:%.+]] = "onnx.Pad"([[VAR_34_]], [[VAR_35_]], [[VAR_36_]], [[VAR_37_]]) {mode = "constant"} : (tensor<1x2x7x3xf32>, tensor<8xi64>, none, none) -> tensor<1x2x7x3xf32>
// CHECK:           onnx.Return [[VAR_38_]] : tensor<1x2x7x3xf32>
  }

// -----

// Test output_padding. Additional pads are inserted after Conv op

  func.func @test_convtrans_outputpadding(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x10x8xf32> {
// CHECK-LABEL:  func.func @test_convtrans_outputpadding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>)

    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, output_shape = [10, 8], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x10x8xf32>
    onnx.Return %1 : tensor<1x2x10x8xf32>

// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x2x3x3xf32>) -> tensor<3x3x1x2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.ReverseSequence"([[VAR_1_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReverseSequence"([[VAR_3_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x2xf32>) -> tensor<1x2x3x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0, 2, 3]} : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_8_]]) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<3xi64>) -> (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>)
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Pad"([[VAR_9_]]#0, [[VAR_10_]], [[VAR_11_]], [[VAR_12_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, none, none) -> tensor<1x1x3x3xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_17_:%.+]] = "onnx.Pad"([[VAR_9_]]#1, [[VAR_14_]], [[VAR_15_]], [[VAR_16_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, none, none) -> tensor<1x1x3x3xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_13_]], [[VAR_17_]], [[VAR_9_]]#2) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x7x3xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]]:3 = "onnx.Split"([[VAR_18_]], [[VAR_19_]]) {axis = 3 : si64} : (tensor<1x1x7x3xf32>, tensor<3xi64>) -> (tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>)
// CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Pad"([[VAR_20_]]#0, [[VAR_21_]], [[VAR_22_]], [[VAR_23_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, none, none) -> tensor<1x1x7x2xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_28_:%.+]] = "onnx.Pad"([[VAR_20_]]#1, [[VAR_25_]], [[VAR_26_]], [[VAR_27_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, none, none) -> tensor<1x1x7x2xf32>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Concat"([[VAR_24_]], [[VAR_28_]], [[VAR_20_]]#2) {axis = 3 : si64} : (tensor<1x1x7x2xf32>, tensor<1x1x7x2xf32>, tensor<1x1x7x1xf32>) -> tensor<1x1x7x5xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Conv"([[VAR_29_]], [[VAR_7_]], [[VAR_0_]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 2, 2, 2]} : (tensor<1x1x7x5xf32>, tensor<2x1x3x3xf32>, none) -> tensor<1x2x9x7xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 1, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Pad"([[VAR_30_]], [[VAR_31_]], [[VAR_32_]], [[VAR_33_]]) {mode = "constant"} : (tensor<1x2x9x7xf32>, tensor<8xi64>, none, none) -> tensor<1x2x10x7xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_36_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_38_:%.+]] = "onnx.Pad"([[VAR_34_]], [[VAR_35_]], [[VAR_36_]], [[VAR_37_]]) {mode = "constant"} : (tensor<1x2x10x7xf32>, tensor<8xi64>, none, none) -> tensor<1x2x10x8xf32>
// CHECK:           onnx.Return [[VAR_38_]] : tensor<1x2x10x8xf32>
  }

// -----

// Test for unknown dimension in spatial dimensions

  func.func @test_convtranspose_unknown_spatial_dim(%arg0: tensor<?x?x3x3xf32>, %arg1: tensor<?x?x3x3xf32>) -> tensor<?x?x10x8xf32> {
// CHECK-LABEL:  func.func @test_convtranspose_unknown_spatial_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3x3xf32>, [[PARAM_1_:%.+]]: tensor<?x?x3x3xf32>)
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "test", output_padding = [1, 1], output_shape = [10, 8], strides = [3, 2]} : (tensor<?x?x3x3xf32>, tensor<?x?x3x3xf32>, none) -> tensor<?x?x10x8xf32>
    onnx.Return %1 : tensor<?x?x10x8xf32>

// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<?x?x3x3xf32>) -> tensor<3x3x?x?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.ReverseSequence"([[VAR_1_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x?x?xf32>, tensor<3xi64>) -> tensor<3x3x?x?xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReverseSequence"([[VAR_3_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x?x?xf32>, tensor<3xi64>) -> tensor<3x3x?x?xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x?x?xf32>) -> tensor<?x?x3x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0, 2, 3]} : (tensor<?x?x3x3xf32>) -> tensor<?x?x3x3xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_8_]]) {axis = 2 : si64} : (tensor<?x?x3x3xf32>, tensor<3xi64>) -> (tensor<?x?x1x3xf32>, tensor<?x?x1x3xf32>, tensor<?x?x1x3xf32>)
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Pad"([[VAR_9_]]#0, [[VAR_10_]], [[VAR_11_]], [[VAR_12_]]) {mode = "constant"} : (tensor<?x?x1x3xf32>, tensor<8xi64>, none, none) -> tensor<?x?x3x3xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_17_:%.+]] = "onnx.Pad"([[VAR_9_]]#1, [[VAR_14_]], [[VAR_15_]], [[VAR_16_]]) {mode = "constant"} : (tensor<?x?x1x3xf32>, tensor<8xi64>, none, none) -> tensor<?x?x3x3xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_13_]], [[VAR_17_]], [[VAR_9_]]#2) {axis = 2 : si64} : (tensor<?x?x3x3xf32>, tensor<?x?x3x3xf32>, tensor<?x?x1x3xf32>) -> tensor<?x?x7x3xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]]:3 = "onnx.Split"([[VAR_18_]], [[VAR_19_]]) {axis = 3 : si64} : (tensor<?x?x7x3xf32>, tensor<3xi64>) -> (tensor<?x?x7x1xf32>, tensor<?x?x7x1xf32>, tensor<?x?x7x1xf32>)
// CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Pad"([[VAR_20_]]#0, [[VAR_21_]], [[VAR_22_]], [[VAR_23_]]) {mode = "constant"} : (tensor<?x?x7x1xf32>, tensor<8xi64>, none, none) -> tensor<?x?x7x2xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_28_:%.+]] = "onnx.Pad"([[VAR_20_]]#1, [[VAR_25_]], [[VAR_26_]], [[VAR_27_]]) {mode = "constant"} : (tensor<?x?x7x1xf32>, tensor<8xi64>, none, none) -> tensor<?x?x7x2xf32>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Concat"([[VAR_24_]], [[VAR_28_]], [[VAR_20_]]#2) {axis = 3 : si64} : (tensor<?x?x7x2xf32>, tensor<?x?x7x2xf32>, tensor<?x?x7x1xf32>) -> tensor<?x?x7x5xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Conv"([[VAR_29_]], [[VAR_7_]], [[VAR_0_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [2, 2, 2, 2]} : (tensor<?x?x7x5xf32>, tensor<?x?x3x3xf32>, none) -> tensor<?x?x9x7xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 1, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Pad"([[VAR_30_]], [[VAR_31_]], [[VAR_32_]], [[VAR_33_]]) {mode = "constant"} : (tensor<?x?x9x7xf32>, tensor<8xi64>, none, none) -> tensor<?x?x10x7xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_36_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_38_:%.+]] = "onnx.Pad"([[VAR_34_]], [[VAR_35_]], [[VAR_36_]], [[VAR_37_]]) {mode = "constant"} : (tensor<?x?x10x7xf32>, tensor<8xi64>, none, none) -> tensor<?x?x10x8xf32>
// CHECK:           onnx.Return [[VAR_38_]] : tensor<?x?x10x8xf32>
  }
