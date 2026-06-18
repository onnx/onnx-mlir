// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// RUN: onnx-mlir-opt --fuse-matmul-add-to-xfe-matmul-bias %s | FileCheck %s

// CHECK-LABEL: @fuse_matmul_add_f32
func.func @fuse_matmul_add_f32(%arg0: tensor<4x8xf32>) -> tensor<4x16xf32> {
  %w = onnx.Constant {value = dense<1.0> : tensor<8x16xf32>} : tensor<8x16xf32>
  %b = onnx.Constant {value = dense<0.5> : tensor<16xf32>} : tensor<16xf32>
  %mm = "onnx.MatMul"(%arg0, %w) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %r = "onnx.Add"(%mm, %b) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}
// CHECK-NOT: onnx.MatMul
// CHECK-NOT: onnx.Add
// CHECK: "onnx.XFEMatMulBias"

// -----

// CHECK-LABEL: @fuse_matmul_add_reversed_operand_order
func.func @fuse_matmul_add_reversed_operand_order(%arg0: tensor<4x8xf32>) -> tensor<4x16xf32> {
  %w = onnx.Constant {value = dense<1.0> : tensor<8x16xf32>} : tensor<8x16xf32>
  %b = onnx.Constant {value = dense<0.25> : tensor<16xf32>} : tensor<16xf32>
  %mm = "onnx.MatMul"(%arg0, %w) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %r = "onnx.Add"(%b, %mm) : (tensor<16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}
// CHECK-NOT: onnx.MatMul
// CHECK-NOT: onnx.Add
// CHECK: "onnx.XFEMatMulBias"

// -----

// MatMul has an extra consumer: must not fuse.
// CHECK-LABEL: @no_fuse_matmul_multiuse
func.func @no_fuse_matmul_multiuse(%arg0: tensor<4x8xf32>) -> (tensor<4x16xf32>, tensor<4x16xf32>) {
  %w = onnx.Constant {value = dense<1.0> : tensor<8x16xf32>} : tensor<8x16xf32>
  %b = onnx.Constant {value = dense<0.5> : tensor<16xf32>} : tensor<16xf32>
  %mm = "onnx.MatMul"(%arg0, %w) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %r = "onnx.Add"(%mm, %b) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  return %r, %mm : tensor<4x16xf32>, tensor<4x16xf32>
}
// CHECK: onnx.MatMul
// CHECK: onnx.Add

// -----

// Per-tensor weight + per-tensor bias [N]: granularity matches -> fuse.
// CHECK-LABEL: @fuse_quant_per_tensor_weight_bias
func.func @fuse_quant_per_tensor_weight_bias(%arg0: tensor<4x8x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<4x3x!quant.uniform<i8:f32, 1.000000e+00>> {
  %w = onnx.Constant {value = dense<1> : tensor<8x3xi8>} : tensor<8x3x!quant.uniform<i8:f32, 2.000000e+00>>
  %b = onnx.Constant {value = dense<0> : tensor<3xi8>} : tensor<3x!quant.uniform<i8:f32, 1.000000e-01:0>>
  %mm = "onnx.MatMul"(%arg0, %w) : (tensor<4x8x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<8x3x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<4x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %r = "onnx.Add"(%mm, %b) : (tensor<4x3x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<3x!quant.uniform<i8:f32, 1.000000e-01:0>>) -> tensor<4x3x!quant.uniform<i8:f32, 1.000000e+00>>
  return %r : tensor<4x3x!quant.uniform<i8:f32, 1.000000e+00>>
}
// CHECK-NOT: onnx.MatMul
// CHECK-NOT: onnx.Add
// CHECK: "onnx.XFEMatMulBias"

// -----

// Per-channel weight + per-channel bias [N]: granularity matches -> fuse.
// CHECK-LABEL: @fuse_quant_per_channel_weight_bias
func.func @fuse_quant_per_channel_weight_bias(%arg0: tensor<4x8x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>> {
  %w = onnx.Constant {value = dense<1> : tensor<8x3xi8>} : tensor<8x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  %b = onnx.Constant {value = dense<0> : tensor<3xi8>} : tensor<3x!quant.uniform<i8:f32:0, {1.100000e-01, 1.200000e-01, 1.300000e-01}>>
  %mm = "onnx.MatMul"(%arg0, %w) : (tensor<4x8x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<8x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  %r = "onnx.Add"(%mm, %b) : (tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>, tensor<3x!quant.uniform<i8:f32:0, {1.100000e-01, 1.200000e-01, 1.300000e-01}>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  return %r : tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
}
// CHECK-NOT: onnx.MatMul
// CHECK-NOT: onnx.Add
// CHECK: "onnx.XFEMatMulBias"

// -----

// Per-channel weight + per-tensor bias: granularity mismatch -> keep separate Add.
// CHECK-LABEL: @no_fuse_per_channel_weight_per_tensor_bias
func.func @no_fuse_per_channel_weight_per_tensor_bias(%arg0: tensor<4x8x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>> {
  %w = onnx.Constant {value = dense<1> : tensor<8x3xi8>} : tensor<8x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  %b = onnx.Constant {value = dense<0> : tensor<3xi8>} : tensor<3x!quant.uniform<i8:f32, 1.000000e-01:0>>
  %mm = "onnx.MatMul"(%arg0, %w) : (tensor<4x8x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<8x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  %r = "onnx.Add"(%mm, %b) : (tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>, tensor<3x!quant.uniform<i8:f32, 1.000000e-01:0>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  return %r : tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
}
// CHECK: onnx.MatMul
// CHECK: onnx.Add

// -----

// 2-D [seq, N] broadcast bias (not effectively 1-D) -> keep separate Add.
// CHECK-LABEL: @no_fuse_2d_broadcast_bias
func.func @no_fuse_2d_broadcast_bias(%arg0: tensor<4x8xf32>) -> tensor<4x16xf32> {
  %w = onnx.Constant {value = dense<1.0> : tensor<8x16xf32>} : tensor<8x16xf32>
  %b = onnx.Constant {value = dense<0.5> : tensor<4x16xf32>} : tensor<4x16xf32>
  %mm = "onnx.MatMul"(%arg0, %w) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %r = "onnx.Add"(%mm, %b) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
  return %r : tensor<4x16xf32>
}
// CHECK: onnx.MatMul
// CHECK: onnx.Add
