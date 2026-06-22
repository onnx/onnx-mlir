// RUN: onnx-mlir-opt --shape-inference --decompose-onnx="enable-convtranspose-phased enable-separate-phased-convs-for-convtranspose" %s -split-input-file | FileCheck %s

// Test that with enable-separate-phased-convs-for-convtranspose, the 4-phase
// kernel 6x6 case (which normally uses a single combined Conv) instead produces
// 4 separate Conv ops, concatenated and then merged back via
// Reshape->Transpose->Reshape.

func.func @test_sep_4phase_kernel_66(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {
  %0 = "onnx.Constant" () { value = dense<0.02> : tensor<256xf32> } : () -> tensor<256xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x20x32xf32>
  onnx.Return %1 : tensor<1x256x20x32xf32>

// CHECK-LABEL:  func.func @test_sep_4phase_kernel_66
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x512x10x16xf32>, [[PARAM_1_:%.+]]: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 256, 20, 32]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 256, 10, 16]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<256xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xf32>) -> tensor<6x6x512x256xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.ReverseSequence"([[VAR_11_]], [[VAR_9_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xf32>, tensor<6xi64>) -> tensor<6x6x512x256xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.ReverseSequence"([[VAR_12_]], [[VAR_9_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xf32>, tensor<6xi64>) -> tensor<6x6x512x256xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_13_]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xf32>) -> tensor<512x256x6x6xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xf32>) -> tensor<256x512x6x6xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_19_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x10x16xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_16_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x10x16xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_17_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x10x16xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_18_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x10x16xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Concat"([[VAR_20_]], [[VAR_22_]], [[VAR_23_]], [[VAR_21_]]) {axis = 1 : si64} : (tensor<1x256x10x16xf32>, tensor<1x256x10x16xf32>, tensor<1x256x10x16xf32>, tensor<1x256x10x16xf32>) -> tensor<1x1024x10x16xf32>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Reshape"([[VAR_24_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1024x10x16xf32>, tensor<5xi64>) -> tensor<2x2x256x10x16xf32>
// CHECK:           [[VAR_26_:%.+]] = "onnx.Transpose"([[VAR_25_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x10x16xf32>) -> tensor<256x10x2x16x2xf32>
// CHECK:           [[VAR_27_:%.+]] = "onnx.Reshape"([[VAR_26_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<256x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x256x20x32xf32>
// CHECK:           onnx.Return [[VAR_27_]] : tensor<1x256x20x32xf32>
// CHECK:         }
}

// -----

// Test kernel 2x2 case (also normally uses a single combined Conv, now forced
// to 4 separate convs, still merged with Reshape->Transpose->Reshape).


func.func @test_sep_4phase_kernel_22(%arg0: tensor<1x64x8x8xf32>, %arg1: tensor<64x32x2x2xf32>) -> tensor<1x32x16x16xf32> {
  %0 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x8x8xf32>, tensor<64x32x2x2xf32>, none) -> tensor<1x32x16x16xf32>
  onnx.Return %1 : tensor<1x32x16x16xf32>

// CHECK-LABEL:  func.func @test_sep_4phase_kernel_22
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x8x8xf32>, [[PARAM_1_:%.+]]: tensor<64x32x2x2xf32>) -> tensor<1x32x16x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 32, 16, 16]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 32, 8, 8]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<64x32x2x2xf32>) -> tensor<2x2x64x32xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.ReverseSequence"([[VAR_9_]], [[VAR_7_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<2x2x64x32xf32>, tensor<2xi64>) -> tensor<2x2x64x32xf32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.ReverseSequence"([[VAR_10_]], [[VAR_7_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<2x2x64x32xf32>, tensor<2xi64>) -> tensor<2x2x64x32xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [2, 3, 0, 1]} : (tensor<2x2x64x32xf32>) -> tensor<64x32x2x2xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_12_]]) {perm = [1, 0, 2, 3]} : (tensor<64x32x2x2xf32>) -> tensor<32x64x2x2xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_5_]], [[VAR_7_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<32x64x2x2xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x64x1x1xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_4_]], [[VAR_7_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<32x64x2x2xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x64x1x1xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_3_]], [[VAR_7_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<32x64x2x2xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x64x1x1xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_2_]], [[VAR_7_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<32x64x2x2xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x64x1x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_17_]], [[VAR_8_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x64x8x8xf32>, tensor<32x64x1x1xf32>, none) -> tensor<1x32x8x8xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_14_]], [[VAR_8_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x64x8x8xf32>, tensor<32x64x1x1xf32>, none) -> tensor<1x32x8x8xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_15_]], [[VAR_8_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x64x8x8xf32>, tensor<32x64x1x1xf32>, none) -> tensor<1x32x8x8xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_16_]], [[VAR_8_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x64x8x8xf32>, tensor<32x64x1x1xf32>, none) -> tensor<1x32x8x8xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Concat"([[VAR_18_]], [[VAR_20_]], [[VAR_21_]], [[VAR_19_]]) {axis = 1 : si64} : (tensor<1x32x8x8xf32>, tensor<1x32x8x8xf32>, tensor<1x32x8x8xf32>, tensor<1x32x8x8xf32>) -> tensor<1x128x8x8xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Reshape"([[VAR_22_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x128x8x8xf32>, tensor<5xi64>) -> tensor<2x2x32x8x8xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Transpose"([[VAR_23_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x32x8x8xf32>) -> tensor<32x8x2x8x2xf32>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Reshape"([[VAR_24_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<32x8x2x8x2xf32>, tensor<4xi64>) -> tensor<1x32x16x16xf32>
// CHECK:           onnx.Return [[VAR_25_]] : tensor<1x32x16x16xf32>
// CHECK:         }
}

// -----

// Test that the kernel 4x4 case (which already uses 4 separate convs) keeps
// producing 4 separate convs merged with Reshape->Transpose->Reshape.


func.func @test_sep_4phase_kernel_44(%arg0: tensor<1x128x10x16xf32>, %arg1: tensor<128x32x4x4xf32>) -> tensor<1x32x20x32xf32> {
  %0 = "onnx.Constant" () { value = dense<0.02> : tensor<32xf32> } : () -> tensor<32xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 4], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x128x10x16xf32>, tensor<128x32x4x4xf32>, tensor<32xf32>) -> tensor<1x32x20x32xf32>
  onnx.Return %1 : tensor<1x32x20x32xf32>

// CHECK-LABEL:  func.func @test_sep_4phase_kernel_44
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x128x10x16xf32>, [[PARAM_1_:%.+]]: tensor<128x32x4x4xf32>) -> tensor<1x32x20x32xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 32, 20, 32]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 32, 10, 16]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<4> : tensor<4xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<32xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<128x32x4x4xf32>) -> tensor<4x4x128x32xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.ReverseSequence"([[VAR_11_]], [[VAR_9_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x4x128x32xf32>, tensor<4xi64>) -> tensor<4x4x128x32xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.ReverseSequence"([[VAR_12_]], [[VAR_9_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x4x128x32xf32>, tensor<4xi64>) -> tensor<4x4x128x32xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_13_]]) {perm = [2, 3, 0, 1]} : (tensor<4x4x128x32xf32>) -> tensor<128x32x4x4xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [1, 0, 2, 3]} : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_19_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_16_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 0, 0], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_17_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 1, 1, 0], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_18_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Concat"([[VAR_21_]], [[VAR_23_]], [[VAR_22_]], [[VAR_20_]]) {axis = 1 : si64} : (tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>) -> tensor<1x128x10x16xf32>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Reshape"([[VAR_24_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x128x10x16xf32>, tensor<5xi64>) -> tensor<2x2x32x10x16xf32>
// CHECK:           [[VAR_26_:%.+]] = "onnx.Transpose"([[VAR_25_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x32x10x16xf32>) -> tensor<32x10x2x16x2xf32>
// CHECK:           [[VAR_27_:%.+]] = "onnx.Reshape"([[VAR_26_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<32x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x32x20x32xf32>
// CHECK:           onnx.Return [[VAR_27_]] : tensor<1x32x20x32xf32>
// CHECK:         }
}

// -----

// Test k3x3 with pads [0,0,1,1] (weights padded to 4x4). With non-uniform
// per-phase padding, conv outputs are correct size directly (no slicing of the
// activations), still merged with Reshape->Transpose->Reshape.


func.func @test_sep_4phase_pads_0011(%arg0: tensor<1x128x10x16xf32>, %arg1: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {
  %0 = "onnx.Constant" () { value = dense<0.02> : tensor<32xf32> } : () -> tensor<32xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], strides = [2, 2]} : (tensor<1x128x10x16xf32>, tensor<128x32x3x3xf32>, tensor<32xf32>) -> tensor<1x32x20x32xf32>
  onnx.Return %1 : tensor<1x32x20x32xf32>

// CHECK-LABEL:  func.func @test_sep_4phase_pads_0011
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x128x10x16xf32>, [[PARAM_1_:%.+]]: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 32, 20, 32]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 32, 10, 16]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 1, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<32xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<128x32x3x3xf32>) -> tensor<3x3x128x32xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.ReverseSequence"([[VAR_14_]], [[VAR_12_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x128x32xf32>, tensor<3xi64>) -> tensor<3x3x128x32xf32>
// CHECK:           [[VAR_16_:%.+]] = "onnx.ReverseSequence"([[VAR_15_]], [[VAR_12_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x128x32xf32>, tensor<3xi64>) -> tensor<3x3x128x32xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Transpose"([[VAR_16_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x128x32xf32>) -> tensor<128x32x3x3xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_17_]]) {perm = [1, 0, 2, 3]} : (tensor<128x32x3x3xf32>) -> tensor<32x128x3x3xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Pad"([[VAR_18_]], [[VAR_11_]], [[VAR_10_]], [[VAR_9_]]) {mode = "constant"} : (tensor<32x128x3x3xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<32x128x4x4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_23_]], [[VAR_13_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_20_]], [[VAR_13_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 0, 0], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_21_]], [[VAR_13_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 1, 1, 0], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_22_]], [[VAR_13_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK:           [[VAR_28_:%.+]] = "onnx.Concat"([[VAR_25_]], [[VAR_27_]], [[VAR_26_]], [[VAR_24_]]) {axis = 1 : si64} : (tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>) -> tensor<1x128x10x16xf32>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Reshape"([[VAR_28_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x128x10x16xf32>, tensor<5xi64>) -> tensor<2x2x32x10x16xf32>
// CHECK:           [[VAR_30_:%.+]] = "onnx.Transpose"([[VAR_29_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x32x10x16xf32>) -> tensor<32x10x2x16x2xf32>
// CHECK:           [[VAR_31_:%.+]] = "onnx.Reshape"([[VAR_30_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<32x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x32x20x32xf32>
// CHECK:           onnx.Return [[VAR_31_]] : tensor<1x32x20x32xf32>
// CHECK:         }
}

// -----

// Test k3x3 with pads [1,1,0,0] (weights padded to 4x4). Same as above
// but with different weight padding direction.


func.func @test_sep_4phase_pads_1100(%arg0: tensor<1x128x10x16xf32>, %arg1: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {
  %0 = "onnx.Constant" () { value = dense<0.02> : tensor<32xf32> } : () -> tensor<32xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 0, 0], strides = [2, 2]} : (tensor<1x128x10x16xf32>, tensor<128x32x3x3xf32>, tensor<32xf32>) -> tensor<1x32x20x32xf32>
  onnx.Return %1 : tensor<1x32x20x32xf32>

// CHECK-LABEL:  func.func @test_sep_4phase_pads_1100
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x128x10x16xf32>, [[PARAM_1_:%.+]]: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 32, 20, 32]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 32, 10, 16]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<[0, 0, 1, 1, 0, 0, 0, 0]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<32xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<128x32x3x3xf32>) -> tensor<3x3x128x32xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.ReverseSequence"([[VAR_14_]], [[VAR_12_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x128x32xf32>, tensor<3xi64>) -> tensor<3x3x128x32xf32>
// CHECK:           [[VAR_16_:%.+]] = "onnx.ReverseSequence"([[VAR_15_]], [[VAR_12_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x128x32xf32>, tensor<3xi64>) -> tensor<3x3x128x32xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Transpose"([[VAR_16_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x128x32xf32>) -> tensor<128x32x3x3xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_17_]]) {perm = [1, 0, 2, 3]} : (tensor<128x32x3x3xf32>) -> tensor<32x128x3x3xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Pad"([[VAR_18_]], [[VAR_11_]], [[VAR_10_]], [[VAR_9_]]) {mode = "constant"} : (tensor<32x128x3x3xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<32x128x4x4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_23_]], [[VAR_13_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_20_]], [[VAR_13_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 0, 0], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_21_]], [[VAR_13_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 1, 1, 0], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_22_]], [[VAR_13_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK:           [[VAR_28_:%.+]] = "onnx.Concat"([[VAR_25_]], [[VAR_27_]], [[VAR_26_]], [[VAR_24_]]) {axis = 1 : si64} : (tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>) -> tensor<1x128x10x16xf32>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Reshape"([[VAR_28_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x128x10x16xf32>, tensor<5xi64>) -> tensor<2x2x32x10x16xf32>
// CHECK:           [[VAR_30_:%.+]] = "onnx.Transpose"([[VAR_29_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x32x10x16xf32>) -> tensor<32x10x2x16x2xf32>
// CHECK:           [[VAR_31_:%.+]] = "onnx.Reshape"([[VAR_30_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<32x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x32x20x32xf32>
// CHECK:           onnx.Return [[VAR_31_]] : tensor<1x32x20x32xf32>
// CHECK:         }
}

// -----

// Test that even with enable-separate-phased-convs-for-convtranspose, a 4-phase
// kernel 6x6 ConvTranspose whose output channels are NOT DMA-aligned (here a
// single output channel) still combines the 4 phased weights into a single
// Conv (weights concatenated to 4x32x3x3), since the separate-conv path is only
// taken when output channels are DMA-aligned. Shapes taken from the asura
// asura_convt_outc1 model. Note the single combined Conv producing 4 channels.

func.func @test_sep_4phase_kernel_66_unaligned_oc(%arg0: tensor<1x32x80x336xf32>, %arg1: tensor<32x1x6x6xf32>) -> tensor<1x1x160x672xf32> {
  %0 = "onnx.Constant" () { value = dense<0.02> : tensor<1xf32> } : () -> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x32x80x336xf32>, tensor<32x1x6x6xf32>, tensor<1xf32>) -> tensor<1x1x160x672xf32>
  onnx.Return %1 : tensor<1x1x160x672xf32>

// CHECK-LABEL:  func.func @test_sep_4phase_kernel_66_unaligned_oc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x80x336xf32>, [[PARAM_1_:%.+]]: tensor<32x1x6x6xf32>) -> tensor<1x1x160x672xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 1, 160, 672]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 1, 80, 336]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<32x1x6x6xf32>) -> tensor<6x6x32x1xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.ReverseSequence"([[VAR_11_]], [[VAR_9_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x32x1xf32>, tensor<6xi64>) -> tensor<6x6x32x1xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.ReverseSequence"([[VAR_12_]], [[VAR_9_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x32x1xf32>, tensor<6xi64>) -> tensor<6x6x32x1xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_13_]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x32x1xf32>) -> tensor<32x1x6x6xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [1, 0, 2, 3]} : (tensor<32x1x6x6xf32>) -> tensor<1x32x6x6xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<1x32x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x3x3xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<1x32x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x3x3xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<1x32x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x3x3xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<1x32x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x3x3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Concat"([[VAR_19_]], [[VAR_17_]], [[VAR_18_]], [[VAR_16_]]) {axis = 0 : si64} : (tensor<1x32x3x3xf32>, tensor<1x32x3x3xf32>, tensor<1x32x3x3xf32>, tensor<1x32x3x3xf32>) -> tensor<4x32x3x3xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Concat"([[VAR_10_]], [[VAR_10_]], [[VAR_10_]], [[VAR_10_]]) {axis = 0 : si64} : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_20_]], [[VAR_21_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x32x80x336xf32>, tensor<4x32x3x3xf32>, tensor<4xf32>) -> tensor<1x4x80x336xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Reshape"([[VAR_22_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x4x80x336xf32>, tensor<5xi64>) -> tensor<2x2x1x80x336xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Transpose"([[VAR_23_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x1x80x336xf32>) -> tensor<1x80x2x336x2xf32>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Reshape"([[VAR_24_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x80x2x336x2xf32>, tensor<4xi64>) -> tensor<1x1x160x672xf32>
// CHECK:           onnx.Return [[VAR_25_]] : tensor<1x1x160x672xf32>
// CHECK:         }
}
