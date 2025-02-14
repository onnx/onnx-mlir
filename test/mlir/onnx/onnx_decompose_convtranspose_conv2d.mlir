// RUN: onnx-mlir-opt --shape-inference --decompose-onnx --disable-convtranspose-decompose --enable-convtranspose-decompose-conv  %s -split-input-file | FileCheck %s

// -----

  func.func @test_convtrans_even_kernel(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {    
    %0 = "onnx.Constant" () { value= dense<0.02> : tensor<256xf32>} : ()-> tensor<256xf32>
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x20x32xf32>
    onnx.Return %1 : tensor<1x256x20x32xf32>
  }
  // CHECK-LABEL:   func.func @test_convtrans_even_kernel(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<1x512x10x16xf32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 256, 20, 32]> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 256, 10, 1, 32]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<[1, 256, 10, 16, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<[0, 0, 0, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<[0, 0, 1, 0]> : tensor<4xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<0> : tensor<4xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<[1, 1, 2, 2]> : tensor<4xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<[256, 512, 6, 6]> : tensor<4xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<[0, 0, 1, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<2.000000e-02> : tensor<256xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xf32>) -> tensor<6x6x512x256xf32>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_12]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xf32>, tensor<6xi64>) -> tensor<6x6x512x256xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_12]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xf32>, tensor<6xi64>) -> tensor<6x6x512x256xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xf32>) -> tensor<512x256x6x6xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xf32>) -> tensor<256x512x6x6xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Slice"(%[[VAL_18]], %[[VAL_11]], %[[VAL_10]], %[[VAL_8]], %[[VAL_9]]) : (tensor<256x512x6x6xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Slice"(%[[VAL_18]], %[[VAL_7]], %[[VAL_10]], %[[VAL_8]], %[[VAL_9]]) : (tensor<256x512x6x6xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Slice"(%[[VAL_18]], %[[VAL_6]], %[[VAL_10]], %[[VAL_8]], %[[VAL_9]]) : (tensor<256x512x6x6xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_18]], %[[VAL_5]], %[[VAL_10]], %[[VAL_8]], %[[VAL_9]]) : (tensor<256x512x6x6xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_19]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x10x16xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_20]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<*xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_21]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<*xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_22]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x10x16xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<*xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Reshape"(%[[VAL_23]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x256x10x16xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Reshape"(%[[VAL_24]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Reshape"(%[[VAL_25]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Reshape"(%[[VAL_26]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<5xi64>) -> tensor<1x256x10x16x1xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Concat"(%[[VAL_27]], %[[VAL_29]]) {axis = -1 : si64} : (tensor<1x256x10x16x1xf32>, tensor<1x256x10x16x1xf32>) -> tensor<1x256x10x16x2xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Concat"(%[[VAL_30]], %[[VAL_28]]) {axis = -1 : si64} : (tensor<1x256x10x16x1xf32>, tensor<1x256x10x16x1xf32>) -> tensor<1x256x10x16x2xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Reshape"(%[[VAL_31]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x10x16x2xf32>, tensor<5xi64>) -> tensor<1x256x10x1x32xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Reshape"(%[[VAL_32]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x10x16x2xf32>, tensor<5xi64>) -> tensor<1x256x10x1x32xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Concat"(%[[VAL_33]], %[[VAL_34]]) {axis = -2 : si64} : (tensor<1x256x10x1x32xf32>, tensor<1x256x10x1x32xf32>) -> tensor<1x256x10x2x32xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Reshape"(%[[VAL_35]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x256x10x2x32xf32>, tensor<4xi64>) -> tensor<1x256x20x32xf32>
// CHECK:           onnx.Return %[[VAL_36]] : tensor<1x256x20x32xf32>
// CHECK:         }

  // -----

  func.func @test_convtrans_garuda_odd_kernel(%arg0: tensor<1x128x10x16xf32>, %arg1: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {
    %0 = "onnx.Constant" () { value= dense<0.02> : tensor<32xf32>} : ()-> tensor<32xf32>
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0,0,1,1], strides = [2, 2]} : (tensor<1x128x10x16xf32>, tensor<128x32x3x3xf32>, tensor<32xf32>) -> tensor<1x32x20x32xf32>
    onnx.Return %1 : tensor<1x32x20x32xf32>
  }
// CHECK-LABEL:   func.func @test_convtrans_garuda_odd_kernel(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<1x128x10x16xf32>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: tensor<128x32x3x3xf32>) -> tensor<1x32x20x32xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<2.000000e-02> : tensor<32xf32>
// CHECK:           %[[VAL_3:.*]] = "onnx.ConvTranspose"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0, 0, 1, 1], strides = [2, 2]} : (tensor<1x128x10x16xf32>, tensor<128x32x3x3xf32>, tensor<32xf32>) -> tensor<1x32x20x32xf32>
// CHECK:           onnx.Return %[[VAL_3]] : tensor<1x32x20x32xf32>
// CHECK:         }