// RUN: onnx-mlir-opt --split-input-file %s --convert-matmul-to-xfe-conv | FileCheck %s

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

// CHECK: %[[XFE_CONV:.*]] = "onnx.XFEConv"(%[[RESHAPE1_OUT]], %[[CONV_WEIGHT]], %{{.*}}) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], onnx_node_name = "MatMul_1", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x1x64x!quant.uniform<u8:f32, 2.500000e-01>>, tensor<32x1x1x64x!quant.uniform<u8:f32, 2.500000e-01>>, none) -> tensor<1x1x1x32x!quant.uniform<u8:f32, 2.500000e-01>>

// CHECK: %[[RESHAPE2_OUT:.*]] = "onnx.Reshape"(%[[XFE_CONV]], %[[RESHAPE2_SHAPE]])
// CHECK-SAME: tensor<1x32x!quant.uniform<u8:f32, 2.500000e-01>>

// CHECK: %[[DEQUANT_OUT:.*]] = "onnx.DequantizeLinear"(%[[RESHAPE2_OUT]]
// CHECK-SAME: tensor<1x32xf32>

// CHECK-NOT: "onnx.MatMul"

// -----

//===----------------------------------------------------------------------===//
// Gemm with transB=1 and bias 
// B is [N, K] — should be reshaped directly, no transpose needed.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @gemm_transB1_with_bias
func.func @gemm_transB1_with_bias(%arg0: tensor<1x64x!quant.uniform<u8:f32, 0.145:67>>) -> tensor<1x32x!quant.uniform<u8:f32, 1.134>> {
  %B = onnx.Constant {value = dense_resource<__elided__> : tensor<32x64xi8>} : tensor<32x64x!quant.uniform<i8:f32, 0.002>>
  %C = onnx.Constant {value = dense_resource<__elided__> : tensor<32xi32>} : tensor<32x!quant.uniform<i32:f32, 2.81E-4>>
  %0 = "onnx.Gemm"(%arg0, %B, %C) {
    alpha = 1.000000e+00 : f32,
    beta = 1.000000e+00 : f32,
    onnx_node_name = "/fc/Gemm",
    transA = 0 : si64,
    transB = 1 : si64} : (tensor<1x64x!quant.uniform<u8:f32, 0.145:67>>, tensor<32x64x!quant.uniform<i8:f32, 0.002>>, tensor<32x!quant.uniform<i32:f32, 2.81E-4>>) -> tensor<1x32x!quant.uniform<u8:f32, 1.134>>
  return %0 : tensor<1x32x!quant.uniform<u8:f32, 1.134>>
}
// CHECK-DAG: %[[RESHAPE1_SHAPE:.*]] = onnx.Constant dense<[1, 1, 1, 64]>
// CHECK-DAG: %[[WEIGHT_SHAPE:.*]] = onnx.Constant dense<[32, 1, 1, 64]>
// CHECK-DAG: %[[RESHAPE2_SHAPE:.*]] = onnx.Constant dense<[1, 32]>

// transB=1: B is [N, K]=[32, 64], reshaped directly to [32, 1, 1, 64] — no Transpose
// CHECK: %[[RESHAPE1_OUT:.*]] = "onnx.Reshape"(%arg0, %[[RESHAPE1_SHAPE]])
// CHECK-SAME: tensor<1x1x1x64x!quant.uniform<u8:f32, 1.450000e-01:67>>
// CHECK-NOT: "onnx.Transpose"
// CHECK: %[[CONV_WEIGHT:.*]] = "onnx.Reshape"(%{{.*}}, %[[WEIGHT_SHAPE]])
// CHECK-SAME: tensor<32x1x1x64x!quant.uniform<i8:f32, 2.000000e-03>>
// CHECK: %[[XFE_CONV:.*]] = "onnx.XFEConv"(%[[RESHAPE1_OUT]], %[[CONV_WEIGHT]], %{{.*}}) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], onnx_node_name = "/fc/Gemm", pads = [0, 0, 0, 0], strides = [1, 1]}
// CHECK-SAME: tensor<1x1x1x32x!quant.uniform<u8:f32, 1.134000e+00>>
// CHECK: %[[RESHAPE2_OUT:.*]] = "onnx.Reshape"(%[[XFE_CONV]], %[[RESHAPE2_SHAPE]])
// CHECK-SAME: tensor<1x32x!quant.uniform<u8:f32, 1.134000e+00>>
// CHECK-NOT: "onnx.Gemm"

// -----

//===----------------------------------------------------------------------===//
// Gemm with transB=0 (standard: B is [K, N], requires Transpose)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @gemm_transB0
func.func @gemm_transB0(%arg0: tensor<1x64x!quant.uniform<u8:f32, 0.25>>) -> tensor<1x32x!quant.uniform<u8:f32, 0.25>> {
  %B = onnx.Constant {value = dense_resource<__elided__> : tensor<64x32xi8>} : tensor<64x32x!quant.uniform<i8:f32, 0.002>>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %B, %none) {
    alpha = 1.000000e+00 : f32,
    beta = 1.000000e+00 : f32,
    onnx_node_name = "/fc/Gemm_0",
    transA = 0 : si64,
    transB = 0 : si64} : (tensor<1x64x!quant.uniform<u8:f32, 0.25>>, tensor<64x32x!quant.uniform<i8:f32, 0.002>>, none) -> tensor<1x32x!quant.uniform<u8:f32, 0.25>>
  return %0 : tensor<1x32x!quant.uniform<u8:f32, 0.25>>
}
// CHECK-DAG: %[[RESHAPE1_SHAPE:.*]] = onnx.Constant dense<[1, 1, 1, 64]>
// CHECK-DAG: %[[WEIGHT_SHAPE:.*]] = onnx.Constant dense<[32, 1, 1, 64]>
// CHECK-DAG: %[[RESHAPE2_SHAPE:.*]] = onnx.Constant dense<[1, 32]>

// transB=0: B is [K, N]=[64, 32], needs Transpose to [32, 64] then Reshape
// CHECK: %[[RESHAPE1_OUT:.*]] = "onnx.Reshape"(%arg0, %[[RESHAPE1_SHAPE]])
// CHECK-SAME: tensor<1x1x1x64x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK: %[[TRANSPOSED_B:.*]] = "onnx.Transpose"
// CHECK-SAME: tensor<32x64x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CONV_WEIGHT:.*]] = "onnx.Reshape"(%[[TRANSPOSED_B]], %[[WEIGHT_SHAPE]])
// CHECK-SAME: tensor<32x1x1x64x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[XFE_CONV:.*]] = "onnx.XFEConv"(%[[RESHAPE1_OUT]], %[[CONV_WEIGHT]], %{{.*}}) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], onnx_node_name = "/fc/Gemm_0", pads = [0, 0, 0, 0], strides = [1, 1]}
// CHECK-SAME: tensor<1x1x1x32x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK: %[[RESHAPE2_OUT:.*]] = "onnx.Reshape"(%[[XFE_CONV]], %[[RESHAPE2_SHAPE]])
// CHECK-SAME: tensor<1x32x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK-NOT: "onnx.Gemm"

// -----

//===----------------------------------------------------------------------===//
// Gemm with transB=1, K=29696 ( model exact shape)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @gemm_transB1_large_K_not_converted
func.func @gemm_transB1_large_K_not_converted(%arg0: tensor<1x29696x!quant.uniform<u8:f32, 0.14517389237880707:67>>) -> tensor<1x128x!quant.uniform<u8:f32, 1.133776068687439>> {
  %B = onnx.Constant {value = dense_resource<__elided__> : tensor<128x29696xi8>} : tensor<128x29696x!quant.uniform<i8:f32, 0.0019356828415766358>>
  %C = onnx.Constant {value = dense_resource<__elided__> : tensor<128xi32>} : tensor<128x!quant.uniform<i32:f32, 2.810106088872999E-4>>
  %0 = "onnx.Gemm"(%arg0, %B, %C) {
    alpha = 1.000000e+00 : f32,
    beta = 1.000000e+00 : f32,
    onnx_node_name = "/fc_face/fc_face.0/Gemm",
    transA = 0 : si64,
    transB = 1 : si64} : (tensor<1x29696x!quant.uniform<u8:f32, 0.14517389237880707:67>>, tensor<128x29696x!quant.uniform<i8:f32, 0.0019356828415766358>>, tensor<128x!quant.uniform<i32:f32, 2.810106088872999E-4>>) -> tensor<1x128x!quant.uniform<u8:f32, 1.133776068687439>>
  return %0 : tensor<1x128x!quant.uniform<u8:f32, 1.133776068687439>>
}
// Gemm must survive — K=29696 cannot be decomposed into valid H*W*C
// CHECK: "onnx.Gemm"
// CHECK-NOT: "onnx.XFEConv"

// -----

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

// -----

//===----------------------------------------------------------------------===//
// MatMul with per-axis quantized weight
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_per_axis_quant_weight
// Weight [K=4, N=2] with per-axis quant on axis 1 (N = output features).
// After transpose to [N, K] = [2, 4], per-axis dim should become 0.
func.func @matmul_per_axis_quant_weight(%arg0: tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x2x!quant.uniform<u8:f32, 0.1:128>> {
  %w = onnx.Constant {value = dense<1> : tensor<4x2xi8>} : tensor<4x2x!quant.uniform<i8:f32:1, {0.05, 0.06}>>
  %0 = "onnx.MatMul"(%arg0, %w) : (tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>, tensor<4x2x!quant.uniform<i8:f32:1, {0.05, 0.06}>>) -> tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>
  return %0 : tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>
}
// Per-axis dim 1 (N) should remap to axis 0 in the conv weight [N, 1, 1, K]
// CHECK: tensor<2x1x1x4x!quant.uniform<i8:f32:0, {5.000000e-02,6.000000e-02}>>
// CHECK: onnx.XFEConv
// CHECK-NOT: onnx.MatMul
