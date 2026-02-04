// RUN: flexml-opt --split-input-file %s --convert-matmul-to-xfe-conv | FileCheck %s

// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// MatMul -> Reshape -> XFEConv -> Reshape Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_to_xfe_conv_quantized
// Test basic MatMul to XFEConv conversion with quantized types
func.func @matmul_to_xfe_conv_quantized(%arg0: tensor<1x64xf32> {onnx.name = "input"}) -> (tensor<1x32xf32> {onnx.name = "output"}) {
  %0 = onnx.Constant {value = dense_resource<__elided__> : tensor<64x32xi8>} : tensor<64x32x!quant.uniform<u8:f32, 2.500000e-01>>
  %1 = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.QuantizeLinear"(%arg0, %1, %2) {
    axis = 1 : si64,
    block_size = 0 : si64,
    onnx_node_name = "QuantizeLinear_1",
    output_dtype = 0 : si64,
    saturate = 1 : si64} : (tensor<1x64xf32>, tensor<f32>, tensor<i8>) -> tensor<1x64x!quant.uniform<u8:f32, 2.500000e-01>>
  %4 = "onnx.MatMul"(%3, %0) {onnx_node_name = "MatMul_1"} : (tensor<1x64x!quant.uniform<u8:f32, 2.500000e-01>>, tensor<64x32x!quant.uniform<u8:f32, 2.500000e-01>>) -> tensor<1x32x!quant.uniform<u8:f32, 2.500000e-01>>
  %5 = "onnx.DequantizeLinear"(%4, %1, %2) {
    axis = 1 : si64,
    block_size = 0 : si64,
    onnx_node_name = "DequantizeLinear_2"} : (tensor<1x32x!quant.uniform<u8:f32, 2.500000e-01>>, tensor<f32>, tensor<i8>) -> tensor<1x32xf32>
  return %5 : tensor<1x32xf32>
}
// CHECK-DAG: %[[RESHAPE1_SHAPE:.*]] = onnx.Constant dense<[1, 1, 1, 64]>
// CHECK-DAG: %[[WEIGHT_SHAPE:.*]] = onnx.Constant dense<[32, 1, 1, 64]>
// CHECK-DAG: %[[RESHAPE2_SHAPE:.*]] = onnx.Constant dense<[1, 32]>

// CHECK: %[[QUANT_INPUT:.*]] = "onnx.QuantizeLinear"
// CHECK-SAME: tensor<1x64x!quant.uniform<u8:f32, 2.500000e-01>>

// CHECK: %[[RESHAPE1_OUT:.*]] = "onnx.Reshape"(%[[QUANT_INPUT]], %[[RESHAPE1_SHAPE]])
// CHECK-SAME: tensor<1x1x1x64x!quant.uniform<u8:f32, 2.500000e-01>>

// CHECK: %[[CONV_WEIGHT:.*]] = "onnx.Reshape"(%{{.*}}, %[[WEIGHT_SHAPE]])
// CHECK-SAME: tensor<32x1x1x64x!quant.uniform<u8:f32, 2.500000e-01>>

// CHECK: %[[XFE_CONV:.*]] = "onnx.XFEConv"(%[[RESHAPE1_OUT]], %[[CONV_WEIGHT]], %{{.*}}) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x1x64x!quant.uniform<u8:f32, 2.500000e-01>>, tensor<32x1x1x64x!quant.uniform<u8:f32, 2.500000e-01>>, none) -> tensor<1x1x1x32x!quant.uniform<u8:f32, 2.500000e-01>>

// CHECK: %[[RESHAPE2_OUT:.*]] = "onnx.Reshape"(%[[XFE_CONV]], %[[RESHAPE2_SHAPE]])
// CHECK-SAME: tensor<1x32x!quant.uniform<u8:f32, 2.500000e-01>>

// CHECK: %[[DEQUANT_OUT:.*]] = "onnx.DequantizeLinear"(%[[RESHAPE2_OUT]]
// CHECK-SAME: tensor<1x32xf32>

// CHECK-NOT: "onnx.MatMul"

//===----------------------------------------------------------------------===//
// MatMul with larger batch size
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_to_xfe_conv_batch
func.func @matmul_to_xfe_conv_batch(%arg0: tensor<4x64x!quant.uniform<u8:f32, 2.500000e-01>>) -> tensor<4x32x!quant.uniform<u8:f32, 2.500000e-01>> {
  %0 = onnx.Constant {value = dense_resource<__elided__> : tensor<64x32xi8>} : tensor<64x32x!quant.uniform<u8:f32, 2.500000e-01>>
  %1 = "onnx.MatMul"(%arg0, %0) : (tensor<4x64x!quant.uniform<u8:f32, 2.500000e-01>>, tensor<64x32x!quant.uniform<u8:f32, 2.500000e-01>>) -> tensor<4x32x!quant.uniform<u8:f32, 2.500000e-01>>
  return %1 : tensor<4x32x!quant.uniform<u8:f32, 2.500000e-01>>
}
// CHECK: %[[RESHAPE1:.*]] = "onnx.Reshape"
// CHECK-SAME: tensor<4x1x1x64x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK: %[[XFE_CONV:.*]] = "onnx.XFEConv"
// CHECK-SAME: tensor<4x1x1x32x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK: %[[RESHAPE2:.*]] = "onnx.Reshape"
// CHECK-SAME: tensor<4x32x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK-NOT: "onnx.MatMul"
