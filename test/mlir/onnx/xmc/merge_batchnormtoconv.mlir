// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: flexml-opt %configPassStx -annotate-config="library-metadata-dirs=%S" %s -merge-batchnorm-to-conv -o - | FileCheck %s

// CHECK-LABEL: @test_merge_batchnorm_to_conv

  func.func @test_merge_batchnorm_to_conv(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x16x2xf32> {
    %0 = onnx.Constant dense<[1.000000e+00, 1.500000e+00]> : tensor<2xf32>
    %1 = onnx.Constant dense<[5.000000e-01, 6.000000e-01]> : tensor<2xf32>
    %2 = onnx.Constant dense<[1.000000e-01, 2.000000e-01]> : tensor<2xf32>
    %3 = onnx.Constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
    %4 = onnx.Constant dense<[1, 2, 16]> : tensor<3xi64>
    %5 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %6 = onnx.Constant dense<0> : tensor<i8>
    %7 = onnx.Constant dense<3.000000e-01> : tensor<f32>
    %8 = onnx.Constant dense_resource<__elided__> : tensor<2x3x3x3xf32>
    %9 = onnx.Constant dense<0.000000e+00> : tensor<2xf32>
    %10 = "onnx.QuantizeLinear"(%arg0, %5, %6) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x3x4x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x3x4x4x!quant.uniform<u8:f32, 0.10000000149011612>>
    %11 = "onnx.Conv"(%10, %8, %9) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x3x4x4x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<2x3x3x3xf32>, tensor<2xf32>) -> tensor<1x2x4x4x!quant.uniform<u8:f32, 0.20000000298023224>>
    %12 = "onnx.Reshape"(%11, %4) {allowzero = 0 : si64} : (tensor<1x2x4x4x!quant.uniform<u8:f32, 0.20000000298023224>>, tensor<3xi64>) -> tensor<1x2x16x!quant.uniform<u8:f32, 0.20000000298023224>>
    %13 = "onnx.Transpose"(%12) {perm = [0, 2, 1]} : (tensor<1x2x16x!quant.uniform<u8:f32, 0.20000000298023224>>) -> tensor<1x16x2x!quant.uniform<u8:f32, 0.20000000298023224>>
    %Y, %running_mean, %running_var = "onnx.BatchNormalization"(%13, %3, %2, %1, %0) {epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32, training_mode = 0 : si64} : (tensor<1x16x2x!quant.uniform<u8:f32, 0.20000000298023224>>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<1x16x2x!quant.uniform<u8:f32, 0.30000001192092896>>, tensor<2xf32>, tensor<2xf32>)
    %14 = "onnx.DequantizeLinear"(%Y, %7, %6) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x16x2x!quant.uniform<u8:f32, 0.30000001192092896>>, tensor<f32>, tensor<i8>) -> tensor<1x16x2xf32>



    return %14 : tensor<1x16x2xf32>
  }
      // CHECK: %[[QUANT:.*]] = "onnx.QuantizeLinear"
    // CHECK: %[[CONV:.*]] = "onnx.Conv"(%[[QUANT]]
    // CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[CONV]]
    // CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[RESHAPE]]
    // CHECK-NOT: onnx.BatchNormalization
    // CHECK: %[[DEQUANT:.*]] = "onnx.DequantizeLinear"(%[[TRANSPOSE]]
    // CHECK: return %[[DEQUANT]]
