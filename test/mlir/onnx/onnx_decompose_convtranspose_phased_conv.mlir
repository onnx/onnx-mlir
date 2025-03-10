// RUN: onnx-mlir-opt --shape-inference --decompose-onnx --enable-convtranspose-decompose-phased-conv %s -split-input-file | FileCheck %s

func.func @test_convtrans_4phase_kernel_shape_66(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<256xf32>} : ()-> tensor<256xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x20x32xf32>
  onnx.Return %1 : tensor<1x256x20x32xf32>

// CHECK-LABEL:   func.func @test_convtrans_4phase_kernel_shape_66(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<1x512x10x16xf32>,
// CHECK-SAME:                                                     %[[VAL_1:.*]]: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 256, 20, 32]> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 256, 10, 1, 32]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<[1, 256, 10, 16, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<7> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[6, 7]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<[7, 6]> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<2.000000e-02> : tensor<256xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xf32>) -> tensor<6x6x512x256xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.ReverseSequence"(%[[VAL_17]], %[[VAL_15]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xf32>, tensor<6xi64>) -> tensor<6x6x512x256xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.ReverseSequence"(%[[VAL_18]], %[[VAL_15]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xf32>, tensor<6xi64>) -> tensor<6x6x512x256xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xf32>) -> tensor<512x256x6x6xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Transpose"(%[[VAL_20]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xf32>) -> tensor<256x512x6x6xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_21]], %[[VAL_12]], %[[VAL_11]], %[[VAL_14]], %[[VAL_13]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Slice"(%[[VAL_21]], %[[VAL_10]], %[[VAL_9]], %[[VAL_14]], %[[VAL_13]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Slice"(%[[VAL_21]], %[[VAL_8]], %[[VAL_7]], %[[VAL_14]], %[[VAL_13]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Slice"(%[[VAL_21]], %[[VAL_6]], %[[VAL_5]], %[[VAL_14]], %[[VAL_13]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_25]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x10x16xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_22]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x10x16xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_23]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x10x16xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_24]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x10x16xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Reshape"(%[[VAL_26]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x256x10x16xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Reshape"(%[[VAL_27]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x256x10x16xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Reshape"(%[[VAL_28]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x256x10x16xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Reshape"(%[[VAL_29]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x256x10x16xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Concat"(%[[VAL_30]], %[[VAL_32]]) {axis = -1 : si64} : (tensor<1x256x10x16x1xf32>, tensor<1x256x10x16x1xf32>) -> tensor<1x256x10x16x2xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Concat"(%[[VAL_33]], %[[VAL_31]]) {axis = -1 : si64} : (tensor<1x256x10x16x1xf32>, tensor<1x256x10x16x1xf32>) -> tensor<1x256x10x16x2xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Reshape"(%[[VAL_34]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x10x16x2xf32>, tensor<5xi64>) -> tensor<1x256x10x1x32xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Reshape"(%[[VAL_35]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x10x16x2xf32>, tensor<5xi64>) -> tensor<1x256x10x1x32xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Concat"(%[[VAL_36]], %[[VAL_37]]) {axis = -2 : si64} : (tensor<1x256x10x1x32xf32>, tensor<1x256x10x1x32xf32>) -> tensor<1x256x10x2x32xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Reshape"(%[[VAL_38]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x256x10x2x32xf32>, tensor<4xi64>) -> tensor<1x256x20x32xf32>
// CHECK:           onnx.Return %[[VAL_39]] : tensor<1x256x20x32xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_4phase_kernel_shape_66_nobias(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {      
  %0 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, none) -> tensor<1x256x20x32xf32>
  onnx.Return %1 : tensor<1x256x20x32xf32>

// CHECK-LABEL:   func.func @test_convtrans_4phase_kernel_shape_66_nobias(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: tensor<1x512x10x16xf32>,
// CHECK-SAME:                                                            %[[VAL_1:.*]]: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 256, 20, 32]> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 256, 10, 1, 32]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<[1, 256, 10, 16, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<7> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[6, 7]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<[7, 6]> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK:           %[[VAL_16:.*]] = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xf32>) -> tensor<6x6x512x256xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.ReverseSequence"(%[[VAL_17]], %[[VAL_15]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xf32>, tensor<6xi64>) -> tensor<6x6x512x256xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.ReverseSequence"(%[[VAL_18]], %[[VAL_15]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xf32>, tensor<6xi64>) -> tensor<6x6x512x256xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xf32>) -> tensor<512x256x6x6xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Transpose"(%[[VAL_20]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xf32>) -> tensor<256x512x6x6xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_21]], %[[VAL_12]], %[[VAL_11]], %[[VAL_14]], %[[VAL_13]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Slice"(%[[VAL_21]], %[[VAL_10]], %[[VAL_9]], %[[VAL_14]], %[[VAL_13]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Slice"(%[[VAL_21]], %[[VAL_8]], %[[VAL_7]], %[[VAL_14]], %[[VAL_13]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Slice"(%[[VAL_21]], %[[VAL_6]], %[[VAL_5]], %[[VAL_14]], %[[VAL_13]]) : (tensor<256x512x6x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_25]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, none) -> tensor<1x256x10x16xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_22]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, none) -> tensor<1x256x10x16xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_23]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, none) -> tensor<1x256x10x16xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_24]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, none) -> tensor<1x256x10x16xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Reshape"(%[[VAL_26]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x256x10x16xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Reshape"(%[[VAL_27]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x256x10x16xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Reshape"(%[[VAL_28]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x256x10x16xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Reshape"(%[[VAL_29]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x256x10x16xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Concat"(%[[VAL_30]], %[[VAL_32]]) {axis = -1 : si64} : (tensor<1x256x10x16x1xf32>, tensor<1x256x10x16x1xf32>) -> tensor<1x256x10x16x2xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Concat"(%[[VAL_33]], %[[VAL_31]]) {axis = -1 : si64} : (tensor<1x256x10x16x1xf32>, tensor<1x256x10x16x1xf32>) -> tensor<1x256x10x16x2xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Reshape"(%[[VAL_34]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x10x16x2xf32>, tensor<5xi64>) -> tensor<1x256x10x1x32xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Reshape"(%[[VAL_35]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x10x16x2xf32>, tensor<5xi64>) -> tensor<1x256x10x1x32xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Concat"(%[[VAL_36]], %[[VAL_37]]) {axis = -2 : si64} : (tensor<1x256x10x1x32xf32>, tensor<1x256x10x1x32xf32>) -> tensor<1x256x10x2x32xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Reshape"(%[[VAL_38]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x256x10x2x32xf32>, tensor<4xi64>) -> tensor<1x256x20x32xf32>
// CHECK:           onnx.Return %[[VAL_39]] : tensor<1x256x20x32xf32>
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

// CHECK-LABEL:   func.func @test_convtrans_9phase(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<1x1x18x74xf32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<1x1x3x3xf32>) -> tensor<1x1x54x222xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<5> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[4, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<[3, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<[5, 4]> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<[3, 4]> : tensor<2xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_17:.*]] = onnx.Constant dense<[5, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_18:.*]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_19:.*]] = onnx.Constant dense<[4, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_20:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_21:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_22:.*]] = onnx.Constant dense<3> : tensor<2xi64>
// CHECK:           %[[VAL_23:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_24:.*]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           %[[VAL_25:.*]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x3x3xf32>) -> tensor<3x3x1x1xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.ReverseSequence"(%[[VAL_26]], %[[VAL_24]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x1xf32>, tensor<3xi64>) -> tensor<3x3x1x1xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.ReverseSequence"(%[[VAL_27]], %[[VAL_24]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x1xf32>, tensor<3xi64>) -> tensor<3x3x1x1xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Transpose"(%[[VAL_28]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x1xf32>) -> tensor<1x1x3x3xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Transpose"(%[[VAL_29]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_22]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_20]], %[[VAL_19]], %[[VAL_23]], %[[VAL_22]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_18]], %[[VAL_17]], %[[VAL_23]], %[[VAL_22]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_16]], %[[VAL_15]], %[[VAL_23]], %[[VAL_22]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_14]], %[[VAL_13]], %[[VAL_23]], %[[VAL_22]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_12]], %[[VAL_11]], %[[VAL_23]], %[[VAL_22]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_10]], %[[VAL_9]], %[[VAL_23]], %[[VAL_22]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_8]], %[[VAL_7]], %[[VAL_23]], %[[VAL_22]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_6]], %[[VAL_5]], %[[VAL_23]], %[[VAL_22]]) : (tensor<1x1x3x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_39]], %[[VAL_25]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_36]], %[[VAL_25]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_42:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_37]], %[[VAL_25]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_43:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_38]], %[[VAL_25]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_44:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_35]], %[[VAL_25]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_45:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_32]], %[[VAL_25]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_46:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_33]], %[[VAL_25]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_47:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_34]], %[[VAL_25]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_48:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_31]], %[[VAL_25]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_49:.*]] = "onnx.Reshape"(%[[VAL_40]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_50:.*]] = "onnx.Reshape"(%[[VAL_41]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_51:.*]] = "onnx.Reshape"(%[[VAL_42]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_52:.*]] = "onnx.Reshape"(%[[VAL_43]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_53:.*]] = "onnx.Reshape"(%[[VAL_44]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_54:.*]] = "onnx.Reshape"(%[[VAL_45]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_55:.*]] = "onnx.Reshape"(%[[VAL_46]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_56:.*]] = "onnx.Reshape"(%[[VAL_47]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_57:.*]] = "onnx.Reshape"(%[[VAL_48]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_58:.*]] = "onnx.Concat"(%[[VAL_49]], %[[VAL_50]], %[[VAL_55]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_59:.*]] = "onnx.Concat"(%[[VAL_52]], %[[VAL_53]], %[[VAL_54]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_60:.*]] = "onnx.Concat"(%[[VAL_51]], %[[VAL_56]], %[[VAL_57]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_61:.*]] = "onnx.Reshape"(%[[VAL_58]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_62:.*]] = "onnx.Reshape"(%[[VAL_59]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_63:.*]] = "onnx.Reshape"(%[[VAL_60]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_64:.*]] = "onnx.Concat"(%[[VAL_61]], %[[VAL_62]], %[[VAL_63]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CHECK:           %[[VAL_65:.*]] = "onnx.Reshape"(%[[VAL_64]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CHECK:           onnx.Return %[[VAL_65]] : tensor<1x1x54x222xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_4phase_pads_0011(%arg0: tensor<1x128x10x16xf32>, %arg1: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<32xf32>} : ()-> tensor<32xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0, 0, 1, 1], strides = [2, 2]} : (tensor<1x128x10x16xf32>, tensor<128x32x3x3xf32>, tensor<32xf32>) -> tensor<1x32x20x32xf32>
  onnx.Return %1 : tensor<1x32x20x32xf32>

// CHECK-LABEL:   func.func @test_convtrans_4phase_pads_0011(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<1x128x10x16xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 32, 20, 32]> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 32, 10, 1, 32]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<[1, 32, 10, 16, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<[10, 18]> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<[12, 16]> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[10, 16]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[12, 18]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<5> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<[4, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<[5, 4]> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_17:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_18:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_19:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_20:.*]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_21:.*]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 1, 1]> : tensor<8xi64>
// CHECK:           %[[VAL_22:.*]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           %[[VAL_23:.*]] = onnx.Constant dense<2.000000e-02> : tensor<32xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<128x32x3x3xf32>) -> tensor<3x3x128x32xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.ReverseSequence"(%[[VAL_24]], %[[VAL_22]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x128x32xf32>, tensor<3xi64>) -> tensor<3x3x128x32xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.ReverseSequence"(%[[VAL_25]], %[[VAL_22]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x128x32xf32>, tensor<3xi64>) -> tensor<3x3x128x32xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Transpose"(%[[VAL_26]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x128x32xf32>) -> tensor<128x32x3x3xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Transpose"(%[[VAL_27]]) {perm = [1, 0, 2, 3]} : (tensor<128x32x3x3xf32>) -> tensor<32x128x3x3xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Pad"(%[[VAL_28]], %[[VAL_21]], %[[VAL_20]], %[[VAL_19]]) {mode = "constant"} : (tensor<32x128x3x3xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<32x128x4x4xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_16]], %[[VAL_15]], %[[VAL_18]], %[[VAL_17]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_14]], %[[VAL_13]], %[[VAL_18]], %[[VAL_17]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_12]], %[[VAL_11]], %[[VAL_18]], %[[VAL_17]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_10]], %[[VAL_9]], %[[VAL_18]], %[[VAL_17]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_33]], %[[VAL_23]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x11x17xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_30]], %[[VAL_23]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x11x17xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_31]], %[[VAL_23]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x11x17xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_32]], %[[VAL_23]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x11x17xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Slice"(%[[VAL_34]], %[[VAL_10]], %[[VAL_8]], %[[VAL_18]], %[[VAL_10]]) : (tensor<1x32x11x17xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x10x16xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Slice"(%[[VAL_35]], %[[VAL_16]], %[[VAL_7]], %[[VAL_18]], %[[VAL_10]]) : (tensor<1x32x11x17xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x10x16xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.Slice"(%[[VAL_36]], %[[VAL_14]], %[[VAL_6]], %[[VAL_18]], %[[VAL_10]]) : (tensor<1x32x11x17xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x10x16xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Slice"(%[[VAL_37]], %[[VAL_12]], %[[VAL_5]], %[[VAL_18]], %[[VAL_10]]) : (tensor<1x32x11x17xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x10x16xf32>
// CHECK:           %[[VAL_42:.*]] = "onnx.Reshape"(%[[VAL_38]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x32x10x16xf32>, tensor<5xi64>) -> tensor<1x32x10x16x1xf32>
// CHECK:           %[[VAL_43:.*]] = "onnx.Reshape"(%[[VAL_39]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x32x10x16xf32>, tensor<5xi64>) -> tensor<1x32x10x16x1xf32>
// CHECK:           %[[VAL_44:.*]] = "onnx.Reshape"(%[[VAL_40]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x32x10x16xf32>, tensor<5xi64>) -> tensor<1x32x10x16x1xf32>
// CHECK:           %[[VAL_45:.*]] = "onnx.Reshape"(%[[VAL_41]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x32x10x16xf32>, tensor<5xi64>) -> tensor<1x32x10x16x1xf32>
// CHECK:           %[[VAL_46:.*]] = "onnx.Concat"(%[[VAL_44]], %[[VAL_42]]) {axis = -1 : si64} : (tensor<1x32x10x16x1xf32>, tensor<1x32x10x16x1xf32>) -> tensor<1x32x10x16x2xf32>
// CHECK:           %[[VAL_47:.*]] = "onnx.Concat"(%[[VAL_43]], %[[VAL_45]]) {axis = -1 : si64} : (tensor<1x32x10x16x1xf32>, tensor<1x32x10x16x1xf32>) -> tensor<1x32x10x16x2xf32>
// CHECK:           %[[VAL_48:.*]] = "onnx.Reshape"(%[[VAL_46]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x32x10x16x2xf32>, tensor<5xi64>) -> tensor<1x32x10x1x32xf32>
// CHECK:           %[[VAL_49:.*]] = "onnx.Reshape"(%[[VAL_47]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x32x10x16x2xf32>, tensor<5xi64>) -> tensor<1x32x10x1x32xf32>
// CHECK:           %[[VAL_50:.*]] = "onnx.Concat"(%[[VAL_49]], %[[VAL_48]]) {axis = -2 : si64} : (tensor<1x32x10x1x32xf32>, tensor<1x32x10x1x32xf32>) -> tensor<1x32x10x2x32xf32>
// CHECK:           %[[VAL_51:.*]] = "onnx.Reshape"(%[[VAL_50]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x32x10x2x32xf32>, tensor<4xi64>) -> tensor<1x32x20x32xf32>
// CHECK:           onnx.Return %[[VAL_51]] : tensor<1x32x20x32xf32>
// CHECK:         }
}

// -----

func.func @test_convtrans_4phase_pads_1100(%arg0: tensor<1x128x10x16xf32>, %arg1: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<32xf32>} : ()-> tensor<32xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1, 1, 0, 0], strides = [2, 2]} : (tensor<1x128x10x16xf32>, tensor<128x32x3x3xf32>, tensor<32xf32>) -> tensor<1x32x20x32xf32>
  onnx.Return %1 : tensor<1x32x20x32xf32>

// CHECK-LABEL:   func.func @test_convtrans_4phase_pads_1100(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<1x128x10x16xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 32, 20, 32]> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 32, 10, 1, 32]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<[1, 32, 10, 16, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<[10, 18]> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<[12, 16]> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[10, 16]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[12, 18]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<5> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<[4, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<[5, 4]> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_17:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_18:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_19:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_20:.*]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_21:.*]] = onnx.Constant dense<[0, 0, 1, 1, 0, 0, 0, 0]> : tensor<8xi64>
// CHECK:           %[[VAL_22:.*]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           %[[VAL_23:.*]] = onnx.Constant dense<2.000000e-02> : tensor<32xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<128x32x3x3xf32>) -> tensor<3x3x128x32xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.ReverseSequence"(%[[VAL_24]], %[[VAL_22]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x128x32xf32>, tensor<3xi64>) -> tensor<3x3x128x32xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.ReverseSequence"(%[[VAL_25]], %[[VAL_22]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x128x32xf32>, tensor<3xi64>) -> tensor<3x3x128x32xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Transpose"(%[[VAL_26]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x128x32xf32>) -> tensor<128x32x3x3xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Transpose"(%[[VAL_27]]) {perm = [1, 0, 2, 3]} : (tensor<128x32x3x3xf32>) -> tensor<32x128x3x3xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Pad"(%[[VAL_28]], %[[VAL_21]], %[[VAL_20]], %[[VAL_19]]) {mode = "constant"} : (tensor<32x128x3x3xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<32x128x4x4xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_16]], %[[VAL_15]], %[[VAL_18]], %[[VAL_17]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_14]], %[[VAL_13]], %[[VAL_18]], %[[VAL_17]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_12]], %[[VAL_11]], %[[VAL_18]], %[[VAL_17]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_10]], %[[VAL_9]], %[[VAL_18]], %[[VAL_17]]) : (tensor<32x128x4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_33]], %[[VAL_23]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x11x17xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_30]], %[[VAL_23]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x11x17xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_31]], %[[VAL_23]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x11x17xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_32]], %[[VAL_23]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x11x17xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Slice"(%[[VAL_34]], %[[VAL_10]], %[[VAL_8]], %[[VAL_18]], %[[VAL_10]]) : (tensor<1x32x11x17xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x10x16xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Slice"(%[[VAL_35]], %[[VAL_16]], %[[VAL_7]], %[[VAL_18]], %[[VAL_10]]) : (tensor<1x32x11x17xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x10x16xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.Slice"(%[[VAL_36]], %[[VAL_14]], %[[VAL_6]], %[[VAL_18]], %[[VAL_10]]) : (tensor<1x32x11x17xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x10x16xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Slice"(%[[VAL_37]], %[[VAL_12]], %[[VAL_5]], %[[VAL_18]], %[[VAL_10]]) : (tensor<1x32x11x17xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x32x10x16xf32>
// CHECK:           %[[VAL_42:.*]] = "onnx.Reshape"(%[[VAL_38]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x32x10x16xf32>, tensor<5xi64>) -> tensor<1x32x10x16x1xf32>
// CHECK:           %[[VAL_43:.*]] = "onnx.Reshape"(%[[VAL_39]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x32x10x16xf32>, tensor<5xi64>) -> tensor<1x32x10x16x1xf32>
// CHECK:           %[[VAL_44:.*]] = "onnx.Reshape"(%[[VAL_40]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x32x10x16xf32>, tensor<5xi64>) -> tensor<1x32x10x16x1xf32>
// CHECK:           %[[VAL_45:.*]] = "onnx.Reshape"(%[[VAL_41]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x32x10x16xf32>, tensor<5xi64>) -> tensor<1x32x10x16x1xf32>
// CHECK:           %[[VAL_46:.*]] = "onnx.Concat"(%[[VAL_44]], %[[VAL_42]]) {axis = -1 : si64} : (tensor<1x32x10x16x1xf32>, tensor<1x32x10x16x1xf32>) -> tensor<1x32x10x16x2xf32>
// CHECK:           %[[VAL_47:.*]] = "onnx.Concat"(%[[VAL_43]], %[[VAL_45]]) {axis = -1 : si64} : (tensor<1x32x10x16x1xf32>, tensor<1x32x10x16x1xf32>) -> tensor<1x32x10x16x2xf32>
// CHECK:           %[[VAL_48:.*]] = "onnx.Reshape"(%[[VAL_46]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x32x10x16x2xf32>, tensor<5xi64>) -> tensor<1x32x10x1x32xf32>
// CHECK:           %[[VAL_49:.*]] = "onnx.Reshape"(%[[VAL_47]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x32x10x16x2xf32>, tensor<5xi64>) -> tensor<1x32x10x1x32xf32>
// CHECK:           %[[VAL_50:.*]] = "onnx.Concat"(%[[VAL_49]], %[[VAL_48]]) {axis = -2 : si64} : (tensor<1x32x10x1x32xf32>, tensor<1x32x10x1x32xf32>) -> tensor<1x32x10x2x32xf32>
// CHECK:           %[[VAL_51:.*]] = "onnx.Reshape"(%[[VAL_50]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x32x10x2x32xf32>, tensor<4xi64>) -> tensor<1x32x20x32xf32>
// CHECK:           onnx.Return %[[VAL_51]] : tensor<1x32x20x32xf32>
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
}