// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file %s --split-group-conv | FileCheck %s

module {
  func.func @test_split_group_conv(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
    // Input quantization parameters
    %input_scale = onnx.Constant dense<5.000000e-02> : tensor<f32>
    %input_zp = onnx.Constant dense<128> : tensor<ui8>

    // Weight constant: 16x4x3x3xui8 = 576 bytes
    // Using dense hex format similar to format_constants.mlir
    %weight_quantized = onnx.Constant {value = dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEEFF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEEFF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F"> : tensor<16x4x3x3xui8>} : tensor<16x4x3x3x!quant.uniform<u8:f32, 0.019999999552965164>>

    // Bias constant: 16xi32 = 64 bytes
    %bias_quantized = onnx.Constant {value = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000"> : tensor<16xi32>} : tensor<16x!quant.uniform<u32:f32, 0.0010000000474974513>>

    // Output quantization parameters
    %output_scale = onnx.Constant dense<1.000000e-01> : tensor<f32>

    // Quantize input
    %quantized_input = "onnx.QuantizeLinear"(%arg0, %input_scale, %input_zp) {
      axis = 1 : si64,
      block_size = 0 : si64,
      output_dtype = 0 : si64,
      saturate = 1 : si64
    } : (tensor<1x16x32x32xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.05000000074505806:128>>

    // Original group conv: group=4, should be split into 2 convs with group=2
    %conv = "onnx.Conv"(%quantized_input, %weight_quantized, %bias_quantized) {
      auto_pad = "NOTSET",
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.05000000074505806:128>>,
         tensor<16x4x3x3x!quant.uniform<u8:f32, 0.019999999552965164>>,
         tensor<16x!quant.uniform<u32:f32, 0.0010000000474974513>>) ->
        tensor<1x16x32x32x!quant.uniform<u8:f32, 0.10000000149011612:128>>

    // Dequantize output
    %output = "onnx.DequantizeLinear"(%conv, %output_scale, %input_zp) {
      axis = 1 : si64,
      block_size = 0 : si64
    } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.10000000149011612:128>>,
         tensor<f32>, tensor<ui8>) -> tensor<1x16x32x32xf32>

    return %output : tensor<1x16x32x32xf32>
  }
}

// CHECK-LABEL: func.func @test_split_group_conv
// CHECK-DAG: %[[QUANT_INPUT:.*]] = "onnx.QuantizeLinear"
// CHECK-DAG: onnx.Constant dense<0>
// CHECK-DAG: onnx.Constant dense<[1, 8, 32, 32]>
// CHECK-DAG: onnx.Constant dense<[0, 8, 0, 0]>
// CHECK-DAG: onnx.Constant dense<[1, 16, 32, 32]>
// CHECK: %[[SLICE1:.*]] = "onnx.Slice"(%[[QUANT_INPUT]]
// CHECK: %[[CONV1:.*]] = "onnx.Conv"(%[[SLICE1]]
// CHECK-SAME: group = 2 : si64
// CHECK-SAME: tensor<8x4x3x3x!quant.uniform<u8:f32, 0.019999999552965164>>
// CHECK-SAME: tensor<8x!quant.uniform<u32:f32, 0.0010000000474974513>>
// CHECK: %[[SLICE2:.*]] = "onnx.Slice"(%[[QUANT_INPUT]]
// CHECK: %[[CONV2:.*]] = "onnx.Conv"(%[[SLICE2]]
// CHECK-SAME: group = 2 : si64
// CHECK-SAME: tensor<8x4x3x3x!quant.uniform<u8:f32, 0.019999999552965164>>
// CHECK-SAME: tensor<8x!quant.uniform<u32:f32, 0.0010000000474974513>>
// CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%[[CONV1]], %[[CONV2]])
// CHECK-SAME: axis = 1 : si64
// CHECK: "onnx.DequantizeLinear"(%[[CONCAT]]
// CHECK: return

// -----

// Test per-axis quantized weights — scales/zeroPoints must be sliced
// 16 output channels, group=4, 4 channels per group, split into 2 convs.
// splitCount = inputChannels / (2 * weightChannels) = 16 / (2*4) = 2
// Per-axis scales on weights: 16 scales split into 2 groups of 8.
module {
  func.func @test_split_group_conv_per_axis(%arg0: tensor<1x16x8x8x!quant.uniform<u8:f32, 0.05:128>>) -> tensor<1x16x8x8x!quant.uniform<u8:f32, 0.1:128>> {
    %weight = onnx.Constant {value = dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEEFF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEEFF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F"> : tensor<16x4x3x3xui8>} : tensor<16x4x3x3x!quant.uniform<u8:f32:0, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16}>>
    %bias = onnx.Constant {value = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000"> : tensor<16xi32>} : tensor<16x!quant.uniform<i32:f32:0, {0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016}>>

    %conv = "onnx.Conv"(%arg0, %weight, %bias) {
      auto_pad = "NOTSET",
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x16x8x8x!quant.uniform<u8:f32, 0.05:128>>,
         tensor<16x4x3x3x!quant.uniform<u8:f32:0, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16}>>,
         tensor<16x!quant.uniform<i32:f32:0, {0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016}>>) ->
        tensor<1x16x8x8x!quant.uniform<u8:f32, 0.1:128>>

    return %conv : tensor<1x16x8x8x!quant.uniform<u8:f32, 0.1:128>>
  }
}

// CHECK-LABEL: func.func @test_split_group_conv_per_axis
// Split into 2 convs: first 8 channels, then next 8 channels
// First split: weight scales {0.01..0.08}, bias scales {0.001..0.008}
// CHECK: onnx.Conv
// CHECK-SAME: tensor<8x4x3x3x!quant.uniform<u8:f32:0, {1.000000e-02,2.000000e-02,3.000000e-02,4.000000e-02,5.000000e-02,6.000000e-02,7.000000e-02,8.000000e-02}>>
// CHECK-SAME: tensor<8x!quant.uniform<i32:f32:0, {1.000000e-03,2.000000e-03,3.000000e-03,4.000000e-03,5.000000e-03,6.000000e-03,7.000000e-03,8.000000e-03}>>
// Second split: weight scales {0.09..0.16}, bias scales {0.009..0.016}
// CHECK: onnx.Conv
// CHECK-SAME: tensor<8x4x3x3x!quant.uniform<u8:f32:0, {{{.*}}1.000000e-01,1.100000e-01,1.200000e-01,1.300000e-01,1.400000e-01,1.500000e-01,1.600000e-01}>>
// CHECK-SAME: tensor<8x!quant.uniform<i32:f32:0, {{{.*}}1.000000e-02,1.100000e-02,1.200000e-02,1.300000e-02,1.400000e-02,1.500000e-02,1.600000e-02}>>
// CHECK: onnx.Concat
