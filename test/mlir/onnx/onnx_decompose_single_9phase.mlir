// RUN: onnx-mlir-opt --shape-inference --decompose-onnx --enable-convtranspose-decompose-4conv %s -split-input-file | FileCheck %s

// -----

  func.func @test_convtrans_9phase(%arg0: tensor<1x1x18x74xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<1x1x54x222xf32> {    
    %0 = "onnx.Constant" () { value= dense<0.02> : tensor<1xf32>} : ()-> tensor<1xf32>
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3,3], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [0,0,0,0], strides = [3,3]} : (tensor<1x1x18x74xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x54x222xf32>
    onnx.Return %1 : tensor<1x1x54x222xf32>
  }

  