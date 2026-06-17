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
// M=4 goes into spatial W: [1, 1, 4, 64]
// CHECK: %[[RESHAPE1:.*]] = "onnx.Reshape"
// CHECK-SAME: tensor<1x1x4x64x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK: %[[XFE_CONV:.*]] = "onnx.XFEConv"
// CHECK-SAME: tensor<1x1x4x32x!quant.uniform<u8:f32, 2.500000e-01>>
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

// -----

//===----------------------------------------------------------------------===//
// MatMul + Add (bias) fusion: quantized
// MatMul followed by Add with constant bias should fuse into XFEConv with bias.
// Bias is re-quantized into conv accumulation domain (int32).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_add_bias_fusion_quantized
func.func @matmul_add_bias_fusion_quantized(
    %arg0: tensor<1x4x!quant.uniform<u16:f32, 2.5E-4:30000>>
) -> tensor<1x2x!quant.uniform<u16:f32, 4.5E-4:25000>> {
  %weight = onnx.Constant {value = dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi8>} : tensor<4x2x!quant.uniform<i8:f32, 5.0E-3>>
  %bias = onnx.Constant {value = dense<[100, 200]> : tensor<2xi16>} : tensor<2x!quant.uniform<u16:f32, 4.5E-5:40>>
  %mm = "onnx.MatMul"(%arg0, %weight) {onnx_node_name = "qkv_matmul"} :
      (tensor<1x4x!quant.uniform<u16:f32, 2.5E-4:30000>>,
       tensor<4x2x!quant.uniform<i8:f32, 5.0E-3>>)
      -> tensor<1x2x!quant.uniform<u16:f32, 4.5E-4:25000>>
  %out = "onnx.Add"(%mm, %bias) :
      (tensor<1x2x!quant.uniform<u16:f32, 4.5E-4:25000>>,
       tensor<2x!quant.uniform<u16:f32, 4.5E-5:40>>)
      -> tensor<1x2x!quant.uniform<u16:f32, 4.5E-4:25000>>
  return %out : tensor<1x2x!quant.uniform<u16:f32, 4.5E-4:25000>>
}
// Bias should be fused into XFEConv as 3rd operand (not none), re-quantized to i32
// bias_scale=4.5E-5, x_scale=2.5E-4, w_scale=5.0E-3 => accumScale=1.25E-6
// biasMul = round(4.5E-5 / 1.25E-6) = 36
// new_bias[0] = (100 - 40) * 36 = 2160
// new_bias[1] = (200 - 40) * 36 = 5760
// Re-quantized bias constant: dense<[2160, 5760]> with i32 quant type
// CHECK: onnx.Constant {value = dense<[2160, 5760]>
// CHECK-SAME: tensor<2x!quant.uniform<i32:f32, 1.250000e-06>>
// CHECK: onnx.XFEConv
// CHECK-SAME: onnx_node_name = "qkv_matmul"
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.MatMul

// -----

//===----------------------------------------------------------------------===//
// MatMul + Add (bias) fusion: float (no re-quantization needed)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_add_bias_fusion_float
func.func @matmul_add_bias_fusion_float(
    %arg0: tensor<4x64xf32>
) -> tensor<4x32xf32> {
  %weight = onnx.Constant {value = dense_resource<__elided__> : tensor<64x32xf32>} : tensor<64x32xf32>
  %bias = onnx.Constant dense<1.0> : tensor<32xf32>
  %mm = "onnx.MatMul"(%arg0, %weight) {onnx_node_name = "fc_matmul"} :
      (tensor<4x64xf32>, tensor<64x32xf32>) -> tensor<4x32xf32>
  %out = "onnx.Add"(%mm, %bias) :
      (tensor<4x32xf32>, tensor<32xf32>) -> tensor<4x32xf32>
  return %out : tensor<4x32xf32>
}
// Bias should be fused as 3rd operand (float, no re-quantization)
// CHECK: onnx.XFEConv
// CHECK-SAME: onnx_node_name = "fc_matmul"
// CHECK-SAME: tensor<32xf32>
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.MatMul

// -----

//===----------------------------------------------------------------------===//
// MatMul without Add should NOT have bias (regression guard)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_no_add_no_bias
func.func @matmul_no_add_no_bias(
    %arg0: tensor<4x64x!quant.uniform<u8:f32, 0.25>>
) -> tensor<4x32x!quant.uniform<u8:f32, 0.25>> {
  %weight = onnx.Constant {value = dense_resource<__elided__> : tensor<64x32xi8>} : tensor<64x32x!quant.uniform<u8:f32, 0.25>>
  %mm = "onnx.MatMul"(%arg0, %weight) :
      (tensor<4x64x!quant.uniform<u8:f32, 0.25>>,
       tensor<64x32x!quant.uniform<u8:f32, 0.25>>)
      -> tensor<4x32x!quant.uniform<u8:f32, 0.25>>
  return %mm : tensor<4x32x!quant.uniform<u8:f32, 0.25>>
}
// No Add follows MatMul — conv bias should be none
// CHECK: onnx.XFEConv
// CHECK-SAME: none)
// CHECK-NOT: onnx.MatMul

// -----

//===----------------------------------------------------------------------===//
// MatMul + Add bias fusion: per-channel weight, per-tensor bias
// Uses scale[0] from weight for accumScale (matches golden behavior).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_add_bias_perchannel_weight
func.func @matmul_add_bias_perchannel_weight(
    %arg0: tensor<1x4x!quant.uniform<u16:f32, 0.01:100>>
) -> tensor<1x2x!quant.uniform<u16:f32, 0.05:200>> {
  %weight = onnx.Constant {value = dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi8>} : tensor<4x2x!quant.uniform<i8:f32:1, {0.002, 0.004}>>
  %bias = onnx.Constant {value = dense<[500, 600]> : tensor<2xi16>} : tensor<2x!quant.uniform<u16:f32, 1.0E-4:50>>
  %mm = "onnx.MatMul"(%arg0, %weight) {onnx_node_name = "pc_matmul"} :
      (tensor<1x4x!quant.uniform<u16:f32, 0.01:100>>,
       tensor<4x2x!quant.uniform<i8:f32:1, {0.002, 0.004}>>)
      -> tensor<1x2x!quant.uniform<u16:f32, 0.05:200>>
  %out = "onnx.Add"(%mm, %bias) :
      (tensor<1x2x!quant.uniform<u16:f32, 0.05:200>>,
       tensor<2x!quant.uniform<u16:f32, 1.0E-4:50>>)
      -> tensor<1x2x!quant.uniform<u16:f32, 0.05:200>>
  return %out : tensor<1x2x!quant.uniform<u16:f32, 0.05:200>>
}
// accumScale = inputScale * weightScale[0] = 0.01 * 0.002 = 2e-5
// biasScale = 1e-4, biasZP = 50
// new_bias[0] = round((500 - 50) * 1e-4 / 2e-5) = round(2250.0) = 2250
// new_bias[1] = round((600 - 50) * 1e-4 / 2e-5) = round(2750.0) = 2750
// Re-quantized bias: accumScale = 0.01 * 0.002 = 2e-5 (uses weightScale[0])
// CHECK: onnx.Constant {value = dense<[2250, 2750]>
// CHECK-SAME: tensor<2x!quant.uniform<i32:f32, 2.000000e-05>>
// CHECK: onnx.XFEConv
// CHECK-SAME: onnx_node_name = "pc_matmul"
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.MatMul

// -----

//===----------------------------------------------------------------------===//
// (f1) Former-NHWC recovery (golden find_former_nhwc): the MatMul input is a
// flatten of a 4-D NHWC tensor [1,7,7,16] -> [1,784], and the contraction
// K (784) == H*W*C (7*7*16). The flattening Reshape is consumed and the conv is
// fed the 4-D producer (%arg0) directly, as a "global" conv with
// kernel = stride = [7,7] and output [1,1,1,32].
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_former_nhwc_recovery
func.func @matmul_former_nhwc_recovery(
    %arg0: tensor<1x7x7x16x!quant.uniform<u8:f32, 0.25>>
) -> tensor<1x32x!quant.uniform<u8:f32, 0.25>> {
  %shape = onnx.Constant dense<[1, 784]> : tensor<2xi64>
  %r = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} :
      (tensor<1x7x7x16x!quant.uniform<u8:f32, 0.25>>, tensor<2xi64>)
      -> tensor<1x784x!quant.uniform<u8:f32, 0.25>>
  %weight = onnx.Constant {value = dense_resource<__elided__> : tensor<784x32xi8>} : tensor<784x32x!quant.uniform<u8:f32, 0.25>>
  %mm = "onnx.MatMul"(%r, %weight) {onnx_node_name = "fc_matmul"} :
      (tensor<1x784x!quant.uniform<u8:f32, 0.25>>,
       tensor<784x32x!quant.uniform<u8:f32, 0.25>>)
      -> tensor<1x32x!quant.uniform<u8:f32, 0.25>>
  return %mm : tensor<1x32x!quant.uniform<u8:f32, 0.25>>
}
// Conv weight is reshaped to [32, 7, 7, 16] (K = 7*7*16 = 784).
// CHECK: onnx.Constant dense<[32, 7, 7, 16]>
// The conv is fed %arg0 directly (the [1,784] reshape is consumed), global
// kernel = stride = [7,7], output [1,1,1,32].
// CHECK: "onnx.XFEConv"(%arg0,
// CHECK-SAME: kernel_shape = [7, 7]
// CHECK-SAME: strides = [7, 7]
// CHECK-SAME: -> tensor<1x1x1x32x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK: "onnx.Reshape"
// CHECK-SAME: -> tensor<1x32x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK-NOT: onnx.MatMul

// -----

//===----------------------------------------------------------------------===//
// (f1b) Leading-flatten pointwise recovery: the 2-D input is a flatten over the
// leading (spatial) dims of a 4-D NHWC tensor [1,8,8,16] -> [64,16]. The matmul
// contracts channels (K=16=last dim, M=64=1*8*8), so the conv reuses the 8x8
// spatial as a 1x1 conv on [1,8,8,16] -> [1,8,8,32] (matches golden is_3dim
// reuse), instead of re-factoring M=64. The flatten Reshape is consumed.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @matmul_leading_flatten_pointwise
func.func @matmul_leading_flatten_pointwise(
    %arg0: tensor<1x8x8x16x!quant.uniform<u8:f32, 0.25>>
) -> tensor<64x32x!quant.uniform<u8:f32, 0.25>> {
  %shape = onnx.Constant dense<[64, 16]> : tensor<2xi64>
  %r = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} :
      (tensor<1x8x8x16x!quant.uniform<u8:f32, 0.25>>, tensor<2xi64>)
      -> tensor<64x16x!quant.uniform<u8:f32, 0.25>>
  %weight = onnx.Constant {value = dense_resource<__elided__> : tensor<16x32xi8>} : tensor<16x32x!quant.uniform<u8:f32, 0.25>>
  %mm = "onnx.MatMul"(%r, %weight) {onnx_node_name = "pw_matmul"} :
      (tensor<64x16x!quant.uniform<u8:f32, 0.25>>,
       tensor<16x32x!quant.uniform<u8:f32, 0.25>>)
      -> tensor<64x32x!quant.uniform<u8:f32, 0.25>>
  return %mm : tensor<64x32x!quant.uniform<u8:f32, 0.25>>
}
// Conv fed %arg0 directly (flatten consumed), 1x1 kernel, output [1,8,8,32].
// CHECK: "onnx.XFEConv"(%arg0,
// CHECK-SAME: kernel_shape = [1, 1]
// CHECK-SAME: strides = [1, 1]
// CHECK-SAME: -> tensor<1x8x8x32x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK: "onnx.Reshape"
// CHECK-SAME: -> tensor<64x32x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK-NOT: onnx.MatMul
