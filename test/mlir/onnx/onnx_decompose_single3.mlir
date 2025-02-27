// RUN: onnx-mlir-opt --shape-inference --decompose-onnx --enable-convtranspose-decompose-4conv %s -split-input-file | FileCheck %s

// -----

  func.func @test_convtrans_even_kernel(%arg0: tensor<1x512x10x16xf32>, %arg1: tensor<512x256x6x6xf32>) -> tensor<1x256x20x32xf32> {    
    %0 = "onnx.Constant" () { value= dense<0.02> : tensor<256xf32>} : ()-> tensor<256xf32>
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], onnx_node_name = "share_proto/decoder_deconv_os16_deconv/BiasAdd", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x10x16xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x20x32xf32>
    onnx.Return %1 : tensor<1x256x20x32xf32>
  }

  