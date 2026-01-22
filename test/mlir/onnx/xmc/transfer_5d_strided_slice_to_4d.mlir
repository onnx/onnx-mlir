// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: flexml-opt %configPassStx -annotate-config="library-metadata-dirs=%S" %s -transfer-5d-strided-slice-to-4d -o - | FileCheck %s

// CHECK-LABEL: func.func @test_quantized_5d_slice_to_4d
func.func @test_quantized_5d_slice_to_4d(%arg0: tensor<1x8x16x32x64xf32>) -> tensor<1x8x16x10x64xf32> {
  // Constants for original 5D slice
  %0 = onnx.Constant dense<1> : tensor<5xi64>
  %1 = onnx.Constant dense<[0, 1, 2, 3, 4]> : tensor<5xi64>
  %2 = onnx.Constant dense<[1, 8, 16, 10, 64]> : tensor<5xi64>
  %3 = onnx.Constant dense<0> : tensor<5xi64>
  %4 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %5 = onnx.Constant dense<0> : tensor<i8>

  %6 = "onnx.QuantizeLinear"(%arg0, %4, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x16x32x64xf32>, tensor<f32>, tensor<i8>) -> tensor<1x8x16x32x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  %7 = "onnx.Slice"(%6, %3, %2, %1, %0) : (tensor<1x8x16x32x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<5xi64>, tensor<5xi64>, tensor<5xi64>, tensor<5xi64>) -> tensor<1x8x16x10x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  %8 = "onnx.DequantizeLinear"(%7, %4, %5) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x16x10x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<f32>, tensor<i8>) -> tensor<1x8x16x10x64xf32>

  return %8 : tensor<1x8x16x10x64xf32>
}
  // CHECK-DAG: %[[STEPS_4D:.*]] = onnx.Constant dense<1> : tensor<4xi64>
  // CHECK-DAG: %[[AXES_4D:.*]] = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  // CHECK-DAG: %[[ENDS_4D:.*]] = onnx.Constant dense<[1, 128, 10, 64]> : tensor<4xi64>
  // CHECK-DAG: %[[STARTS_4D:.*]] = onnx.Constant dense<0> : tensor<4xi64>
  // CHECK-DAG: %[[SHAPE_5D_OUT:.*]] = onnx.Constant dense<[1, 8, 16, 10, 64]> : tensor<5xi64>
  // CHECK-DAG: %[[SHAPE_4D_IN:.*]] = onnx.Constant dense<[1, 128, 32, 64]> : tensor<4xi64>
  // CHECK-DAG: %[[SCALE:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
  // CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<0> : tensor<i8>

  // Quantize input
  // CHECK: %[[QUANT_INPUT:.*]] = "onnx.QuantizeLinear"(%arg0, %[[SCALE]], %[[ZP]])
  // CHECK-SAME: -> tensor<1x8x16x32x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  // Original 5D Slice (should be transformed to 4D)
  // Dims 1 and 2 (8, 16) are fully copied -> collapsed to 128
  // Dim 3 is sliced from 32 to 10

  // CHECK: %[[RESHAPE_IN:.*]] = "onnx.Reshape"(%[[QUANT_INPUT]], %[[SHAPE_4D_IN]])
  // CHECK-SAME: -> tensor<1x128x32x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  // CHECK: %[[SLICE_4D:.*]] = "onnx.Slice"(%[[RESHAPE_IN]], %[[STARTS_4D]], %[[ENDS_4D]], %[[AXES_4D]], %[[STEPS_4D]])
  // CHECK-SAME: -> tensor<1x128x10x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  // CHECK: %[[RESHAPE_OUT:.*]] = "onnx.Reshape"(%[[SLICE_4D]], %[[SHAPE_5D_OUT]])
  // CHECK-SAME: -> tensor<1x8x16x10x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  // Dequantize output
  // CHECK: %[[DEQUANT_OUTPUT:.*]] = "onnx.DequantizeLinear"(%[[RESHAPE_OUT]], %[[SCALE]], %[[ZP]])
  // CHECK-SAME: -> tensor<1x8x16x10x64xf32>

  // CHECK: return %[[DEQUANT_OUTPUT]]
