// RUN: onnx-mlir-opt --shape-inference --decompose-onnx --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test unit strides. Only convert weight tensor

  func.func @test_convtrans_unitstrides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x5x5xf32> attributes {input_names = ["X", "W"], output_names = ["Y"]} {
  // CHECK-LABEL:  func.func @test_convtrans_unitstrides
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>)

    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], output_shape = [5, 5], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x5x5xf32>
    return %1 : tensor<1x2x5x5xf32>

    // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
    // CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x2x3x3xf32>) -> tensor<3x3x1x2xf32>
    // CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.ReverseSequence"([[VAR_1_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
    // CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
    // CHECK:           [[VAR_5_:%.+]] = "onnx.ReverseSequence"([[VAR_3_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
    // CHECK:           [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [3, 2, 0, 1]} : (tensor<3x3x1x2xf32>) -> tensor<2x1x3x3xf32>
    // CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_6_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x1x3x3xf32>, tensor<2x1x3x3xf32>, none) -> tensor<*xf32>
    // CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<0> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Pad"([[VAR_7_]], [[VAR_8_]], [[VAR_9_]]) {mode = "constant"} : (tensor<*xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x2x5x5xf32>
    // CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK:           [[VAR_13_:%.+]] = "onnx.Pad"([[VAR_10_]], [[VAR_11_]], [[VAR_12_]]) {mode = "constant"} : (tensor<1x2x5x5xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x2x5x5xf32>
    // CHECK:           return [[VAR_13_]] : tensor<1x2x5x5xf32>

  }

// -----

// Test 1d input

  func.func @test_convtrans1d_unitstrides(%arg0: tensor<1x1x3xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x5xf32> attributes {input_names = ["X", "W"], output_names = ["Y"]} {
  // CHECK-LABEL:  func.func @test_convtrans1d_unitstrides
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3xf32>)

    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [3], output_shape = [5], pads = [0, 0], strides = [1]} : (tensor<1x1x3xf32>, tensor<1x2x3xf32>, none) -> tensor<1x2x5xf32>
    return %1 : tensor<1x2x5xf32>

    // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
    // CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<1x2x3xf32>) -> tensor<3x1x2xf32>
    // CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
    // CHECK:           [[VAR_3_:%.+]] = "onnx.ReverseSequence"([[VAR_1_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x1x2xf32>, tensor<1xi64>) -> tensor<3x1x2xf32>
    // CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [2, 1, 0]} : (tensor<3x1x2xf32>) -> tensor<2x1x3xf32>
    // CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_4_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [3], pads = [2, 2], strides = [1]} : (tensor<1x1x3xf32>, tensor<2x1x3xf32>, none) -> tensor<*xf32>
    // CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<6xi64>
    // CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK:           [[VAR_8_:%.+]] = "onnx.Pad"([[VAR_5_]], [[VAR_6_]], [[VAR_7_]]) {mode = "constant"} : (tensor<*xf32>, tensor<6xi64>, tensor<f32>) -> tensor<1x2x5xf32>
    // CHECK:           return [[VAR_8_]] : tensor<1x2x5xf32>
  }

// -----

// Test 3d input

  func.func @test_convtrans3d_unitstrides(%arg0: tensor<1x1x3x4x5xf32>, %arg1: tensor<1x2x3x3x3xf32>) -> tensor<1x2x5x6x7xf32> attributes {input_names = ["X", "W"], output_names = ["Y"]} {
  // CHECK-LABEL:  func.func @test_convtrans3d_unitstrides
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x4x5xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3x3xf32>)

    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [3, 3, 3], output_shape = [5, 6, 7], pads = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} : (tensor<1x1x3x4x5xf32>, tensor<1x2x3x3x3xf32>, none) -> tensor<1x2x5x6x7xf32>
    return %1 : tensor<1x2x5x6x7xf32>

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
    // CHECK:           [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [2, 1, 3, 4, 0]} : (tensor<3x1x2x3x3xf32>) -> tensor<2x1x3x3x3xf32>
    // CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_9_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [3, 3, 3], pads = [2, 2, 2, 2, 2, 2], strides = [1, 1, 1]} : (tensor<1x1x3x4x5xf32>, tensor<2x1x3x3x3xf32>, none) -> tensor<*xf32>
    // CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<10xi64>
    // CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Pad"([[VAR_10_]], [[VAR_11_]], [[VAR_12_]]) {mode = "constant"} : (tensor<*xf32>, tensor<10xi64>, tensor<f32>) -> tensor<1x2x5x6x7xf32>
    // CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<0> : tensor<10xi64>
    // CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Pad"([[VAR_13_]], [[VAR_14_]], [[VAR_15_]]) {mode = "constant"} : (tensor<1x2x5x6x7xf32>, tensor<10xi64>, tensor<f32>) -> tensor<1x2x5x6x7xf32>
    // CHECK-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<0> : tensor<10xi64>
    // CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK:           [[VAR_19_:%.+]] = "onnx.Pad"([[VAR_16_]], [[VAR_17_]], [[VAR_18_]]) {mode = "constant"} : (tensor<1x2x5x6x7xf32>, tensor<10xi64>, tensor<f32>) -> tensor<1x2x5x6x7xf32>
    // CHECK:           return [[VAR_19_]] : tensor<1x2x5x6x7xf32>
  }

// -----

// Test non unit strides. Added pads between elements  in input data.

  func.func @test_convtrans_strides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> attributes {input_names = ["X", "W"], output_names = ["Y"]} {
  // CHECK-LABEL:  func.func @test_convtrans_strides
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>)

    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], output_shape = [7, 3], pads = [1, 2, 1, 2], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x7x3xf32>
    return %1 : tensor<1x2x7x3xf32>

    // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
    // CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x2x3x3xf32>) -> tensor<3x3x1x2xf32>
    // CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.ReverseSequence"([[VAR_1_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
    // CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
    // CHECK:           [[VAR_5_:%.+]] = "onnx.ReverseSequence"([[VAR_3_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
    // CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [3, 2, 0, 1]} : (tensor<3x3x1x2xf32>) -> tensor<2x1x3x3xf32>
    // CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_8_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_7_]]) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<3xi64>) -> (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>)
    // CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Pad"([[VAR_8_]]#0, [[VAR_9_]], [[VAR_10_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x1x3x3xf32>
    // CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK:           [[VAR_14_:%.+]] = "onnx.Pad"([[VAR_8_]]#1, [[VAR_12_]], [[VAR_13_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x1x3x3xf32>
    // CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Concat"([[VAR_11_]], [[VAR_14_]], [[VAR_8_]]#2) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x7x3xf32>
    // CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_17_:%.+]]:3 = "onnx.Split"([[VAR_15_]], [[VAR_16_]]) {axis = 3 : si64} : (tensor<1x1x7x3xf32>, tensor<3xi64>) -> (tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>)
    // CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Pad"([[VAR_17_]]#0, [[VAR_18_]], [[VAR_19_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x1x7x2xf32>
    // CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_22_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK:           [[VAR_23_:%.+]] = "onnx.Pad"([[VAR_17_]]#1, [[VAR_21_]], [[VAR_22_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x1x7x2xf32>
    // CHECK:           [[VAR_24_:%.+]] = "onnx.Concat"([[VAR_20_]], [[VAR_23_]], [[VAR_17_]]#2) {axis = 3 : si64} : (tensor<1x1x7x2xf32>, tensor<1x1x7x2xf32>, tensor<1x1x7x1xf32>) -> tensor<1x1x7x5xf32>
    // CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_6_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 0, 1, 0], strides = [1, 1]} : (tensor<1x1x7x5xf32>, tensor<2x1x3x3xf32>, none) -> tensor<*xf32>
    // CHECK-DAG:       [[VAR_26_:%.+]] = onnx.Constant dense<0> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_27_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Pad"([[VAR_25_]], [[VAR_26_]], [[VAR_27_]]) {mode = "constant"} : (tensor<*xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x2x7x3xf32>
    // CHECK-DAG:       [[VAR_29_:%.+]] = onnx.Constant dense<0> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_30_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK:           [[VAR_31_:%.+]] = "onnx.Pad"([[VAR_28_]], [[VAR_29_]], [[VAR_30_]]) {mode = "constant"} : (tensor<1x2x7x3xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x2x7x3xf32>
    // CHECK:           return [[VAR_31_]] : tensor<1x2x7x3xf32>
  }

// -----

// Test output_padding. Additional pads are inserted after Conv op

  func.func @test_convtrans_outputpadding(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x10x8xf32> attributes {input_names = ["X", "W"], output_names = ["Y"]} {
  // CHECK-LABEL:  func.func @test_convtrans_outputpadding
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>)

    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "test", output_padding = [1, 1], output_shape = [10, 8], pads = [0, 0, 0, 0], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x10x8xf32>
    return %1 : tensor<1x2x10x8xf32>

    // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
    // CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x2x3x3xf32>) -> tensor<3x3x1x2xf32>
    // CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.ReverseSequence"([[VAR_1_]], [[VAR_2_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
    // CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
    // CHECK:           [[VAR_5_:%.+]] = "onnx.ReverseSequence"([[VAR_3_]], [[VAR_4_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x2xf32>, tensor<3xi64>) -> tensor<3x3x1x2xf32>
    // CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [3, 2, 0, 1]} : (tensor<3x3x1x2xf32>) -> tensor<2x1x3x3xf32>
    // CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_8_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_7_]]) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<3xi64>) -> (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>)
    // CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Pad"([[VAR_8_]]#0, [[VAR_9_]], [[VAR_10_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x1x3x3xf32>
    // CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 0]> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK:           [[VAR_14_:%.+]] = "onnx.Pad"([[VAR_8_]]#1, [[VAR_12_]], [[VAR_13_]]) {mode = "constant"} : (tensor<1x1x1x3xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x1x3x3xf32>
    // CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Concat"([[VAR_11_]], [[VAR_14_]], [[VAR_8_]]#2) {axis = 2 : si64} : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x7x3xf32>
    // CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_17_:%.+]]:3 = "onnx.Split"([[VAR_15_]], [[VAR_16_]]) {axis = 3 : si64} : (tensor<1x1x7x3xf32>, tensor<3xi64>) -> (tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>, tensor<1x1x7x1xf32>)
    // CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Pad"([[VAR_17_]]#0, [[VAR_18_]], [[VAR_19_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x1x7x2xf32>
    // CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_22_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK:           [[VAR_23_:%.+]] = "onnx.Pad"([[VAR_17_]]#1, [[VAR_21_]], [[VAR_22_]]) {mode = "constant"} : (tensor<1x1x7x1xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x1x7x2xf32>
    // CHECK:           [[VAR_24_:%.+]] = "onnx.Concat"([[VAR_20_]], [[VAR_23_]], [[VAR_17_]]#2) {axis = 3 : si64} : (tensor<1x1x7x2xf32>, tensor<1x1x7x2xf32>, tensor<1x1x7x1xf32>) -> tensor<1x1x7x5xf32>
    // CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_6_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x1x7x5xf32>, tensor<2x1x3x3xf32>, none) -> tensor<*xf32>
    // CHECK-DAG:       [[VAR_26_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 1, 0]> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_27_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Pad"([[VAR_25_]], [[VAR_26_]], [[VAR_27_]]) {mode = "constant"} : (tensor<*xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x2x10x7xf32>
    // CHECK-DAG:       [[VAR_29_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi64>
    // CHECK-DAG:       [[VAR_30_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
    // CHECK:           [[VAR_31_:%.+]] = "onnx.Pad"([[VAR_28_]], [[VAR_29_]], [[VAR_30_]]) {mode = "constant"} : (tensor<1x2x10x7xf32>, tensor<8xi64>, tensor<f32>) -> tensor<1x2x10x8xf32>
    // CHECK:           return [[VAR_31_]] : tensor<1x2x10x8xf32>
  }

