// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-convtranspose-phased %s -split-input-file | FileCheck %s

func.func @test_convtrans_4phase_kernel_shape_66(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<256xf32>} : ()-> tensor<256xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x20x32xf32>
  onnx.Return %1 : tensor<1x256x20x32xf32>

// CHECK-LABEL:  func.func @test_convtrans_4phase_kernel_shape_66
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
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Concat"([[VAR_19_]], [[VAR_17_]], [[VAR_18_]], [[VAR_16_]]) {axis = 0 : si64} : (tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>) -> tensor<1024x512x3x3xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Concat"([[VAR_10_]], [[VAR_10_]], [[VAR_10_]], [[VAR_10_]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_20_]], [[VAR_21_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x10x16xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Reshape"([[VAR_22_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1024x10x16xf32>, tensor<5xi64>) -> tensor<2x2x256x10x16xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Transpose"([[VAR_23_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x10x16xf32>) -> tensor<256x10x2x16x2xf32>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Reshape"([[VAR_24_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<256x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x256x20x32xf32>
// CHECK:           onnx.Return [[VAR_25_]] : tensor<1x256x20x32xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_4phase_kernel_shape_66_nobias(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {      
  %0 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, none) -> tensor<1x256x20x32xf32>
  onnx.Return %1 : tensor<1x256x20x32xf32>

// CHECK-LABEL:  func.func @test_convtrans_4phase_kernel_shape_66_nobias
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
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xf32>) -> tensor<6x6x512x256xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.ReverseSequence"([[VAR_11_]], [[VAR_9_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xf32>, tensor<6xi64>) -> tensor<6x6x512x256xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.ReverseSequence"([[VAR_12_]], [[VAR_9_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xf32>, tensor<6xi64>) -> tensor<6x6x512x256xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_13_]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xf32>) -> tensor<512x256x6x6xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xf32>) -> tensor<256x512x6x6xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Concat"([[VAR_19_]], [[VAR_17_]], [[VAR_18_]], [[VAR_16_]]) {axis = 0 : si64} : (tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>) -> tensor<1024x512x3x3xf32>
// CHECK:           [[VAR_21_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_20_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<1024x512x3x3xf32>, none) -> tensor<1x1024x10x16xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Reshape"([[VAR_21_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1024x10x16xf32>, tensor<5xi64>) -> tensor<2x2x256x10x16xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Transpose"([[VAR_22_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x10x16xf32>) -> tensor<256x10x2x16x2xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Reshape"([[VAR_23_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<256x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x256x20x32xf32>
// CHECK:           onnx.Return [[VAR_24_]] : tensor<1x256x20x32xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_1phase_pads_1111(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
  onnx.Return %1 : tensor<1x1x13x57xf32>

// CHECK-LABEL:   func.func @test_convtrans_1phase_pads_1111(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xf32>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "onnx.ReverseSequence"(%[[VAL_5]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xf32>, tensor<16xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "onnx.ReverseSequence"(%[[VAL_6]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xf32>, tensor<4xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "onnx.Transpose"(%[[VAL_7]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_9:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_10:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_9]], %[[VAL_4]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           onnx.Return %[[VAL_10]] : tensor<1x1x13x57xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_1phase_pads_1100(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x14x58xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1, 1, 0, 0], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x14x58xf32>
  onnx.Return %1 : tensor<1x1x14x58xf32>

// CHECK-LABEL:   func.func @test_convtrans_1phase_pads_1100(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x14x58xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xf32>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "onnx.ReverseSequence"(%[[VAL_5]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xf32>, tensor<16xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "onnx.ReverseSequence"(%[[VAL_6]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xf32>, tensor<4xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "onnx.Transpose"(%[[VAL_7]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_9:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_10:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_9]], %[[VAL_4]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 3, 15], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x14x58xf32>
// CHECK:           onnx.Return %[[VAL_10]] : tensor<1x1x14x58xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_1phase_pads_1000(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x14x59xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x14x59xf32>
  onnx.Return %1 : tensor<1x1x14x59xf32>

// CHECK-LABEL:   func.func @test_convtrans_1phase_pads_1000(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x14x59xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xf32>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "onnx.ReverseSequence"(%[[VAL_5]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xf32>, tensor<16xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "onnx.ReverseSequence"(%[[VAL_6]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xf32>, tensor<4xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "onnx.Transpose"(%[[VAL_7]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_9:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_10:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_9]], %[[VAL_4]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 15, 3, 15], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x14x59xf32>
// CHECK:           onnx.Return %[[VAL_10]] : tensor<1x1x14x59xf32>
// CHECK:         }
}
  
// -----

func.func @test_convtrans_1phase_pads_1010(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x13x59xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1, 0, 1, 0], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x59xf32>
  onnx.Return %1 : tensor<1x1x13x59xf32>

// CHECK-LABEL:   func.func @test_convtrans_1phase_pads_1010(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x59xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xf32>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "onnx.ReverseSequence"(%[[VAL_5]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xf32>, tensor<16xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "onnx.ReverseSequence"(%[[VAL_6]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xf32>, tensor<4xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "onnx.Transpose"(%[[VAL_7]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_9:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_10:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_9]], %[[VAL_4]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 15, 2, 15], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x59xf32>
// CHECK:           onnx.Return %[[VAL_10]] : tensor<1x1x13x59xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_1phase_pads_0000(%arg0: tensor<1x1x27x110xf32>, %arg1: tensor<1x1x2x8xf32>) -> tensor<1x1x28x117xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 8], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x27x110xf32>, tensor<1x1x2x8xf32>, tensor<1xf32>) -> tensor<1x1x28x117xf32>
  onnx.Return %1 : tensor<1x1x28x117xf32>

// CHECK-LABEL:   func.func @test_convtrans_1phase_pads_0000(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<1x1x27x110xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<1x1x2x8xf32>) -> tensor<1x1x28x117xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<8> : tensor<2xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<2> : tensor<8xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x2x8xf32>) -> tensor<2x8x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "onnx.ReverseSequence"(%[[VAL_5]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<2x8x1x1xf32>, tensor<8xi64>) -> tensor<2x8x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "onnx.ReverseSequence"(%[[VAL_6]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<2x8x1x1xf32>, tensor<2xi64>) -> tensor<2x8x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "onnx.Transpose"(%[[VAL_7]]) {perm = [2, 3, 0, 1]} : (tensor<2x8x1x1xf32>) -> tensor<1x1x2x8xf32>
// CHECK:           %[[VAL_9:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x2x8xf32>) -> tensor<1x1x2x8xf32>
// CHECK:           %[[VAL_10:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_9]], %[[VAL_4]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 8], pads = [1, 7, 1, 7], strides = [1, 1]} : (tensor<1x1x27x110xf32>, tensor<1x1x2x8xf32>, tensor<1xf32>) -> tensor<1x1x28x117xf32>
// CHECK:           onnx.Return %[[VAL_10]] : tensor<1x1x28x117xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_9phase(%arg0: tensor<1x1x18x74xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<1x1x54x222xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0, 0, 0, 0], strides = [3, 3]} : (tensor<1x1x18x74xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x54x222xf32>
  onnx.Return %1 : tensor<1x1x54x222xf32>

// CHECK-LABEL:  func.func @test_convtrans_9phase
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x18x74xf32>, [[PARAM_1_:%.+]]: tensor<1x1x3x3xf32>) -> tensor<1x1x54x222xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<3> : tensor<2xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x3x3xf32>) -> tensor<3x3x1x1xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.ReverseSequence"([[VAR_16_]], [[VAR_14_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x1xf32>, tensor<3xi64>) -> tensor<3x3x1x1xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.ReverseSequence"([[VAR_17_]], [[VAR_14_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x1xf32>, tensor<3xi64>) -> tensor<3x3x1x1xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_18_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x1xf32>) -> tensor<1x1x3x3xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Transpose"([[VAR_19_]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_11_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_10_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_9_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_8_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_7_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_6_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_5_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_4_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_3_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_29_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_26_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_27_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_28_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_25_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_22_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_23_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_24_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_21_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_39_:%.+]] = "onnx.Reshape"([[VAR_30_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_40_:%.+]] = "onnx.Reshape"([[VAR_31_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_41_:%.+]] = "onnx.Reshape"([[VAR_32_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_42_:%.+]] = "onnx.Reshape"([[VAR_33_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = "onnx.Reshape"([[VAR_34_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_44_:%.+]] = "onnx.Reshape"([[VAR_35_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_45_:%.+]] = "onnx.Reshape"([[VAR_36_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_46_:%.+]] = "onnx.Reshape"([[VAR_37_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_47_:%.+]] = "onnx.Reshape"([[VAR_38_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = "onnx.Concat"([[VAR_39_]], [[VAR_40_]], [[VAR_45_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_49_:%.+]] = "onnx.Concat"([[VAR_42_]], [[VAR_43_]], [[VAR_44_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_50_:%.+]] = "onnx.Concat"([[VAR_41_]], [[VAR_46_]], [[VAR_47_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_51_:%.+]] = "onnx.Reshape"([[VAR_48_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK-DAG:       [[VAR_52_:%.+]] = "onnx.Reshape"([[VAR_49_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK-DAG:       [[VAR_53_:%.+]] = "onnx.Reshape"([[VAR_50_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           [[VAR_54_:%.+]] = "onnx.Concat"([[VAR_51_]], [[VAR_52_]], [[VAR_53_]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CHECK:           [[VAR_55_:%.+]] = "onnx.Reshape"([[VAR_54_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CHECK:           onnx.Return [[VAR_55_]] : tensor<1x1x54x222xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_4phase_pads_0011(%arg0: tensor<1x128x10x16xf32>, %arg1: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<32xf32>} : ()-> tensor<32xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0, 0, 1, 1], strides = [2, 2]} : (tensor<1x128x10x16xf32>, tensor<128x32x3x3xf32>, tensor<32xf32>) -> tensor<1x32x20x32xf32>
  onnx.Return %1 : tensor<1x32x20x32xf32>

// CHECK-LABEL:  func.func @test_convtrans_4phase_pads_0011
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

func.func @test_convtrans_4phase_pads_1100(%arg0: tensor<1x128x10x16xf32>, %arg1: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<32xf32>} : ()-> tensor<32xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1, 1, 0, 0], strides = [2, 2]} : (tensor<1x128x10x16xf32>, tensor<128x32x3x3xf32>, tensor<32xf32>) -> tensor<1x32x20x32xf32>
  onnx.Return %1 : tensor<1x32x20x32xf32>

// CHECK-LABEL:  func.func @test_convtrans_4phase_pads_1100
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

func.func @test_convtrans_4phase_kernel_dilation2(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x25x37xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<256xf32>} : ()-> tensor<256xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [2, 2], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x25x37xf32>
  onnx.Return %1 : tensor<1x256x25x37xf32>

// CHECK-LABEL:   func.func @test_convtrans_4phase_kernel_dilation2(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: tensor<1x512x10x16xf32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: tensor<512x256x6x6xf32>) -> tensor<1x256x25x37xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<2.000000e-02> : tensor<256xf32>
// CHECK:           %[[VAL_3:.*]] = "onnx.ConvTranspose"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {auto_pad = "NOTSET", dilations = [2, 2], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x25x37xf32>
// CHECK:           onnx.Return %[[VAL_3]] : tensor<1x256x25x37xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_1phase_pads_0000_nodilation(%arg0: tensor<1x1x27x110xf32>, %arg1: tensor<1x1x2x8xf32>) -> tensor<1x1x28x117xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 8], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x27x110xf32>, tensor<1x1x2x8xf32>, tensor<1xf32>) -> tensor<1x1x28x117xf32>
  onnx.Return %1 : tensor<1x1x28x117xf32>

// CHECK-LABEL:   func.func @test_convtrans_1phase_pads_0000_nodilation(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: tensor<1x1x27x110xf32>,
// CHECK-SAME:                                                          %[[VAL_1:.*]]: tensor<1x1x2x8xf32>) -> tensor<1x1x28x117xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<8> : tensor<2xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<2> : tensor<8xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x2x8xf32>) -> tensor<2x8x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "onnx.ReverseSequence"(%[[VAL_5]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<2x8x1x1xf32>, tensor<8xi64>) -> tensor<2x8x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "onnx.ReverseSequence"(%[[VAL_6]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<2x8x1x1xf32>, tensor<2xi64>) -> tensor<2x8x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "onnx.Transpose"(%[[VAL_7]]) {perm = [2, 3, 0, 1]} : (tensor<2x8x1x1xf32>) -> tensor<1x1x2x8xf32>
// CHECK:           %[[VAL_9:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x2x8xf32>) -> tensor<1x1x2x8xf32>
// CHECK:           %[[VAL_10:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_9]], %[[VAL_4]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 8], pads = [1, 7, 1, 7], strides = [1, 1]} : (tensor<1x1x27x110xf32>, tensor<1x1x2x8xf32>, tensor<1xf32>) -> tensor<1x1x28x117xf32>
// CHECK:           onnx.Return %[[VAL_10]] : tensor<1x1x28x117xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_4phase_kernel_shape_22(%arg0: tensor<1x288x8x96xf32>, %arg1: tensor<288x240x2x2xf32>) -> tensor<1x240x16x192xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<240xf32>} : ()-> tensor<240xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2,2], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0,0,0,0], strides = [2, 2]} : (tensor<1x288x8x96xf32>, tensor<288x240x2x2xf32>, tensor<240xf32>) -> tensor<1x240x16x192xf32>
  onnx.Return %1 : tensor<1x240x16x192xf32>

// CHECK-LABEL:  func.func @test_convtrans_4phase_kernel_shape_22
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x288x8x96xf32>, [[PARAM_1_:%.+]]: tensor<288x240x2x2xf32>) -> tensor<1x240x16x192xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 240, 16, 192]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 240, 8, 96]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<240xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<288x240x2x2xf32>) -> tensor<2x2x288x240xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.ReverseSequence"([[VAR_9_]], [[VAR_7_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<2x2x288x240xf32>, tensor<2xi64>) -> tensor<2x2x288x240xf32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.ReverseSequence"([[VAR_10_]], [[VAR_7_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<2x2x288x240xf32>, tensor<2xi64>) -> tensor<2x2x288x240xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [2, 3, 0, 1]} : (tensor<2x2x288x240xf32>) -> tensor<288x240x2x2xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_12_]]) {perm = [1, 0, 2, 3]} : (tensor<288x240x2x2xf32>) -> tensor<240x288x2x2xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_5_]], [[VAR_7_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<240x288x2x2xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<240x288x1x1xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_4_]], [[VAR_7_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<240x288x2x2xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<240x288x1x1xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_3_]], [[VAR_7_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<240x288x2x2xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<240x288x1x1xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_2_]], [[VAR_7_]], [[VAR_6_]], [[VAR_7_]]) : (tensor<240x288x2x2xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<240x288x1x1xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_17_]], [[VAR_15_]], [[VAR_16_]], [[VAR_14_]]) {axis = 0 : si64} : (tensor<240x288x1x1xf32>, tensor<240x288x1x1xf32>, tensor<240x288x1x1xf32>, tensor<240x288x1x1xf32>) -> tensor<960x288x1x1xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Concat"([[VAR_8_]], [[VAR_8_]], [[VAR_8_]], [[VAR_8_]]) {axis = 0 : si64} : (tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> tensor<960xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_18_]], [[VAR_19_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x288x8x96xf32>, tensor<960x288x1x1xf32>, tensor<960xf32>) -> tensor<1x960x8x96xf32>
// CHECK:           [[VAR_21_:%.+]] = "onnx.Reshape"([[VAR_20_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x960x8x96xf32>, tensor<5xi64>) -> tensor<2x2x240x8x96xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Transpose"([[VAR_21_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x240x8x96xf32>) -> tensor<240x8x2x96x2xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Reshape"([[VAR_22_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<240x8x2x96x2xf32>, tensor<4xi64>) -> tensor<1x240x16x192xf32>
// CHECK:           onnx.Return [[VAR_23_]] : tensor<1x240x16x192xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_4phase_kernel_shape_44(%arg0: tensor<1x512x8x8xf32>, %arg1: tensor<512x512x4x4xf32>) -> tensor<1x512x16x16xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<512xf32>} : ()-> tensor<512xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4,4], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1,1,1,1], strides = [2, 2]} : (tensor<1x512x8x8xf32>, tensor<512x512x4x4xf32>, tensor<512xf32>) -> tensor<1x512x16x16xf32>
  onnx.Return %1 : tensor<1x512x16x16xf32>

// CHECK-LABEL:  func.func @test_convtrans_4phase_kernel_shape_44
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x512x8x8xf32>, [[PARAM_1_:%.+]]: tensor<512x512x4x4xf32>) -> tensor<1x512x16x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 512, 16, 16]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 512, 8, 8]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<4> : tensor<4xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<512xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<512x512x4x4xf32>) -> tensor<4x4x512x512xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.ReverseSequence"([[VAR_11_]], [[VAR_9_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x4x512x512xf32>, tensor<4xi64>) -> tensor<4x4x512x512xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.ReverseSequence"([[VAR_12_]], [[VAR_9_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x4x512x512xf32>, tensor<4xi64>) -> tensor<4x4x512x512xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_13_]]) {perm = [2, 3, 0, 1]} : (tensor<4x4x512x512xf32>) -> tensor<512x512x4x4xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [1, 0, 2, 3]} : (tensor<512x512x4x4xf32>) -> tensor<512x512x4x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<512x512x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<512x512x2x2xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<512x512x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<512x512x2x2xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<512x512x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<512x512x2x2xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<512x512x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<512x512x2x2xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_19_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 1, 1], strides = [1, 1]} : (tensor<1x512x8x8xf32>, tensor<512x512x2x2xf32>, tensor<512xf32>) -> tensor<1x512x8x8xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_16_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 0, 0], strides = [1, 1]} : (tensor<1x512x8x8xf32>, tensor<512x512x2x2xf32>, tensor<512xf32>) -> tensor<1x512x8x8xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_17_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 1, 1, 0], strides = [1, 1]} : (tensor<1x512x8x8xf32>, tensor<512x512x2x2xf32>, tensor<512xf32>) -> tensor<1x512x8x8xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_18_]], [[VAR_10_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 1], strides = [1, 1]} : (tensor<1x512x8x8xf32>, tensor<512x512x2x2xf32>, tensor<512xf32>) -> tensor<1x512x8x8xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Concat"([[VAR_21_]], [[VAR_23_]], [[VAR_22_]], [[VAR_20_]]) {axis = 1 : si64} : (tensor<1x512x8x8xf32>, tensor<1x512x8x8xf32>, tensor<1x512x8x8xf32>, tensor<1x512x8x8xf32>) -> tensor<1x2048x8x8xf32>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Reshape"([[VAR_24_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x2048x8x8xf32>, tensor<5xi64>) -> tensor<2x2x512x8x8xf32>
// CHECK:           [[VAR_26_:%.+]] = "onnx.Transpose"([[VAR_25_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x512x8x8xf32>) -> tensor<512x8x2x8x2xf32>
// CHECK:           [[VAR_27_:%.+]] = "onnx.Reshape"([[VAR_26_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<512x8x2x8x2xf32>, tensor<4xi64>) -> tensor<1x512x16x16xf32>
// CHECK:           onnx.Return [[VAR_27_]] : tensor<1x512x16x16xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_4phase_kernel_shape_66_lrelu(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<256xf32>} : ()-> tensor<256xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x20x32xf32>
  %2 = "onnx.LeakyRelu"(%1) {alpha = 1.000000e-01 : f32, onnx_node_name = "LeakyRelu3"} : (tensor<1x256x20x32xf32>) -> tensor<1x256x20x32xf32> 
  onnx.Return %2 : tensor<1x256x20x32xf32>

// CHECK-LABEL:  func.func @test_convtrans_4phase_kernel_shape_66_lrelu
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
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Concat"([[VAR_19_]], [[VAR_17_]], [[VAR_18_]], [[VAR_16_]]) {axis = 0 : si64} : (tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>) -> tensor<1024x512x3x3xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Concat"([[VAR_10_]], [[VAR_10_]], [[VAR_10_]], [[VAR_10_]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_20_]], [[VAR_21_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x10x16xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.LeakyRelu"([[VAR_22_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1024x10x16xf32>) -> tensor<1x1024x10x16xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Reshape"([[VAR_23_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1024x10x16xf32>, tensor<5xi64>) -> tensor<2x2x256x10x16xf32>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Transpose"([[VAR_24_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x10x16xf32>) -> tensor<256x10x2x16x2xf32>
// CHECK:           [[VAR_26_:%.+]] = "onnx.Reshape"([[VAR_25_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<256x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x256x20x32xf32>
// CHECK:           onnx.Return [[VAR_26_]] : tensor<1x256x20x32xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_4phase_kernel_shape_66_relu(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<256xf32>} : ()-> tensor<256xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x20x32xf32>
  %2 = "onnx.Relu"(%1) {onnx_node_name = "Relu3"} : (tensor<1x256x20x32xf32>) -> tensor<1x256x20x32xf32> 
  onnx.Return %2 : tensor<1x256x20x32xf32>

// CHECK-LABEL:  func.func @test_convtrans_4phase_kernel_shape_66_relu
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
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Concat"([[VAR_19_]], [[VAR_17_]], [[VAR_18_]], [[VAR_16_]]) {axis = 0 : si64} : (tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>) -> tensor<1024x512x3x3xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Concat"([[VAR_10_]], [[VAR_10_]], [[VAR_10_]], [[VAR_10_]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_20_]], [[VAR_21_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x10x16xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Relu"([[VAR_22_]]) : (tensor<1x1024x10x16xf32>) -> tensor<1x1024x10x16xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Reshape"([[VAR_23_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1024x10x16xf32>, tensor<5xi64>) -> tensor<2x2x256x10x16xf32>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Transpose"([[VAR_24_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x10x16xf32>) -> tensor<256x10x2x16x2xf32>
// CHECK:           [[VAR_26_:%.+]] = "onnx.Reshape"([[VAR_25_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<256x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x256x20x32xf32>
// CHECK:           onnx.Return [[VAR_26_]] : tensor<1x256x20x32xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_9phase_lrelu(%arg0: tensor<1x1x18x74xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<1x1x54x222xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0, 0, 0, 0], strides = [3, 3]} : (tensor<1x1x18x74xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x54x222xf32>
  %2 = "onnx.LeakyRelu"(%1) {alpha = 1.000000e-01 : f32, onnx_node_name = "LeakyRelu3"} : (tensor<1x1x54x222xf32>) -> tensor<1x1x54x222xf32> 
  onnx.Return %2 : tensor<1x1x54x222xf32>

// CHECK-LABEL:  func.func @test_convtrans_9phase_lrelu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x18x74xf32>, [[PARAM_1_:%.+]]: tensor<1x1x3x3xf32>) -> tensor<1x1x54x222xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<3> : tensor<2xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x3x3xf32>) -> tensor<3x3x1x1xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.ReverseSequence"([[VAR_16_]], [[VAR_14_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x1xf32>, tensor<3xi64>) -> tensor<3x3x1x1xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.ReverseSequence"([[VAR_17_]], [[VAR_14_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x1xf32>, tensor<3xi64>) -> tensor<3x3x1x1xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_18_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x1xf32>) -> tensor<1x1x3x3xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Transpose"([[VAR_19_]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_11_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_10_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_9_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_8_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_7_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_6_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_5_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_4_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_3_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_30_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_29_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.LeakyRelu"([[VAR_30_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_26_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.LeakyRelu"([[VAR_32_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_27_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = "onnx.LeakyRelu"([[VAR_34_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_28_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.LeakyRelu"([[VAR_36_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_25_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_39_:%.+]] = "onnx.LeakyRelu"([[VAR_38_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_40_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_22_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_41_:%.+]] = "onnx.LeakyRelu"([[VAR_40_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_42_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_23_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = "onnx.LeakyRelu"([[VAR_42_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_44_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_24_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_45_:%.+]] = "onnx.LeakyRelu"([[VAR_44_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_46_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_21_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_47_:%.+]] = "onnx.LeakyRelu"([[VAR_46_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = "onnx.Reshape"([[VAR_31_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_49_:%.+]] = "onnx.Reshape"([[VAR_33_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_50_:%.+]] = "onnx.Reshape"([[VAR_35_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_51_:%.+]] = "onnx.Reshape"([[VAR_37_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_52_:%.+]] = "onnx.Reshape"([[VAR_39_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_53_:%.+]] = "onnx.Reshape"([[VAR_41_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_54_:%.+]] = "onnx.Reshape"([[VAR_43_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_55_:%.+]] = "onnx.Reshape"([[VAR_45_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_56_:%.+]] = "onnx.Reshape"([[VAR_47_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_57_:%.+]] = "onnx.Concat"([[VAR_48_]], [[VAR_49_]], [[VAR_54_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_58_:%.+]] = "onnx.Concat"([[VAR_51_]], [[VAR_52_]], [[VAR_53_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_59_:%.+]] = "onnx.Concat"([[VAR_50_]], [[VAR_55_]], [[VAR_56_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_60_:%.+]] = "onnx.Reshape"([[VAR_57_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK-DAG:       [[VAR_61_:%.+]] = "onnx.Reshape"([[VAR_58_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           [[VAR_62_:%.+]] = "onnx.Reshape"([[VAR_59_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           [[VAR_63_:%.+]] = "onnx.Concat"([[VAR_60_]], [[VAR_61_]], [[VAR_62_]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CHECK:           [[VAR_64_:%.+]] = "onnx.Reshape"([[VAR_63_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CHECK:           onnx.Return [[VAR_64_]] : tensor<1x1x54x222xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_1phase_pads_1100_lrelu(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x14x58xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1, 1, 0, 0], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x14x58xf32>
  %2 = "onnx.LeakyRelu"(%1) {alpha = 1.000000e-01 : f32, onnx_node_name = "LeakyRelu3"} : (tensor<1x1x14x58xf32>) -> tensor<1x1x14x58xf32> 
  onnx.Return %2 : tensor<1x1x14x58xf32>

// CHECK-LABEL:   func.func @test_convtrans_1phase_pads_1100_lrelu(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                                     %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x14x58xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xf32>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "onnx.ReverseSequence"(%[[VAL_5]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xf32>, tensor<16xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "onnx.ReverseSequence"(%[[VAL_6]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xf32>, tensor<4xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "onnx.Transpose"(%[[VAL_7]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_9:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_10:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_9]], %[[VAL_4]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 3, 15], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x14x58xf32>
// CHECK:           %[[VAL_11:.*]] = "onnx.LeakyRelu"(%[[VAL_10]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x14x58xf32>) -> tensor<1x1x14x58xf32>
// CHECK:           onnx.Return %[[VAL_11]] : tensor<1x1x14x58xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_1phase_pads_1100_lrelu_default_value(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x14x58xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1, 1, 0, 0], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x14x58xf32>
  %2 = "onnx.LeakyRelu"(%1) {onnx_node_name = "LeakyRelu3"} : (tensor<1x1x14x58xf32>) -> tensor<1x1x14x58xf32> 
  onnx.Return %2 : tensor<1x1x14x58xf32>

// CHECK-LABEL:   func.func @test_convtrans_1phase_pads_1100_lrelu_default_value(
// CHECK-SAME:                                                                   %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                                                   %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x14x58xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xf32>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "onnx.ReverseSequence"(%[[VAL_5]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xf32>, tensor<16xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "onnx.ReverseSequence"(%[[VAL_6]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xf32>, tensor<4xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "onnx.Transpose"(%[[VAL_7]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_9:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_10:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_9]], %[[VAL_4]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 3, 15], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x14x58xf32>
// CHECK:           %[[VAL_11:.*]] = "onnx.LeakyRelu"(%[[VAL_10]]) {alpha = 0.00999999977 : f32} : (tensor<1x1x14x58xf32>) -> tensor<1x1x14x58xf32>
// CHECK:           onnx.Return %[[VAL_11]] : tensor<1x1x14x58xf32>
// CHECK:         }  
}

// -----

func.func @test_convtrans_4phase_kernel_shape_66_lrelu_default_alpha(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<256xf32>} : ()-> tensor<256xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x20x32xf32>
  %2 = "onnx.LeakyRelu"(%1) {onnx_node_name = "LeakyRelu3"} : (tensor<1x256x20x32xf32>) -> tensor<1x256x20x32xf32> 
  onnx.Return %2 : tensor<1x256x20x32xf32>

// CHECK-LABEL:  func.func @test_convtrans_4phase_kernel_shape_66_lrelu_default_alpha
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
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Concat"([[VAR_19_]], [[VAR_17_]], [[VAR_18_]], [[VAR_16_]]) {axis = 0 : si64} : (tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>, tensor<256x512x3x3xf32>) -> tensor<1024x512x3x3xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Concat"([[VAR_10_]], [[VAR_10_]], [[VAR_10_]], [[VAR_10_]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_20_]], [[VAR_21_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x10x16xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.LeakyRelu"([[VAR_22_]]) {alpha = 0.00999999977 : f32} : (tensor<1x1024x10x16xf32>) -> tensor<1x1024x10x16xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Reshape"([[VAR_23_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1024x10x16xf32>, tensor<5xi64>) -> tensor<2x2x256x10x16xf32>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Transpose"([[VAR_24_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x10x16xf32>) -> tensor<256x10x2x16x2xf32>
// CHECK:           [[VAR_26_:%.+]] = "onnx.Reshape"([[VAR_25_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<256x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x256x20x32xf32>
// CHECK:           onnx.Return [[VAR_26_]] : tensor<1x256x20x32xf32>
// CHECK:         }
}
