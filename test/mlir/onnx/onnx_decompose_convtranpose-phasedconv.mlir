// RUN: onnx-mlir-opt --shape-inference --decompose-onnx --enable-convtranspose-decompose-4conv %s -split-input-file | FileCheck %s

// -----

  func.func @test_convtrans_zeropad(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x15x59xf32> {
    %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0,0,0,0], strides = [1,1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>,tensor<1xf32>) -> tensor<1x1x15x59xf32>
    onnx.Return %1 : tensor<1x1x15x59xf32>
  }
// CHECK-LABEL:   func.func @test_convtrans_zeropad(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x15x59xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<2.000000e-02> : tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xf32>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "onnx.ReverseSequence"(%[[VAL_5]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xf32>, tensor<16xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "onnx.ReverseSequence"(%[[VAL_6]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xf32>, tensor<4xi64>) -> tensor<4x16x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "onnx.Transpose"(%[[VAL_7]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_9:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xf32>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_10:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_9]], %[[VAL_4]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [3, 15, 3, 15], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x15x59xf32>
// CHECK:           onnx.Return %[[VAL_10]] : tensor<1x1x15x59xf32>
// CHECK:         }

  // -----

  func.func @test_convtrans_pad1111(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
    %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1,1,1,1], strides = [1,1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>,tensor<1xf32>) -> tensor<1x1x13x57xf32>
    onnx.Return %1 : tensor<1x1x13x57xf32>
  }
// CHECK-LABEL:   func.func @test_convtrans_pad1111(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
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
  // -----

  func.func @test_convtrans_pad_1010(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x13x59xf32> {
    %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1,0,1,0], strides = [1,1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>,tensor<1xf32>) -> tensor<1x1x13x59xf32>
    onnx.Return %1 : tensor<1x1x13x59xf32>
  }
// CHECK-LABEL:   func.func @test_convtrans_pad_1010(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x59xf32> {
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
  // -----

  func.func @test_convtrans_pad1100(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x14x58xf32> {
    %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [1,1,0,0], strides = [1,1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>,tensor<1xf32>) -> tensor<1x1x14x58xf32>
    onnx.Return %1 : tensor<1x1x14x58xf32>
  }
// CHECK-LABEL:   func.func @test_convtrans_pad1100(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x14x58xf32> {
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
  