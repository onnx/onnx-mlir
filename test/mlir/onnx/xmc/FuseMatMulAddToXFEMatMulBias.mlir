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

// Quantized MatMul + per-tensor bias constant (i8 storage).
// CHECK-LABEL: @fuse_quant_per_tensor_bias
func.func @fuse_quant_per_tensor_bias(%arg0: tensor<4x8x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 7.000000e-01, 8.000000e-01}>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>> {
  %w = onnx.Constant {value = dense<1> : tensor<8x3xi8>} : tensor<8x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  %b = onnx.Constant {value = dense<0> : tensor<3xi8>} : tensor<3x!quant.uniform<i8:f32, 1.000000e-01:0>>
  %mm = "onnx.MatMul"(%arg0, %w) : (tensor<4x8x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 7.000000e-01, 8.000000e-01}>>, tensor<8x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  %r = "onnx.Add"(%mm, %b) : (tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>, tensor<3x!quant.uniform<i8:f32, 1.000000e-01:0>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  return %r : tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
}
// CHECK-NOT: onnx.MatMul
// CHECK-NOT: onnx.Add
// CHECK: "onnx.XFEMatMulBias"

// -----

// Quantized bias with per-channel scales (per-axis on broadcast dims);
// folds to rank-1 per-axis axis 0.
// CHECK-LABEL: @fuse_quant_per_axis_bias_broadcast
func.func @fuse_quant_per_axis_bias_broadcast(%arg0: tensor<4x8x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 7.000000e-01, 8.000000e-01}>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>> {
  %w = onnx.Constant {value = dense<1> : tensor<8x3xi8>} : tensor<8x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  %b = onnx.Constant {value = dense<0> : tensor<1x3x1x1xi8>} : tensor<1x3x1x1x!quant.uniform<i8:f32:1, {1.100000e-01, 1.200000e-01, 1.300000e-01}>>
  %mm = "onnx.MatMul"(%arg0, %w) : (tensor<4x8x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 7.000000e-01, 8.000000e-01}>>, tensor<8x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  %r = "onnx.Add"(%mm, %b) : (tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>, tensor<1x3x1x1x!quant.uniform<i8:f32:1, {1.100000e-01, 1.200000e-01, 1.300000e-01}>>) -> tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
  return %r : tensor<4x3x!quant.uniform<i8:f32:1, {1.000000e-01, 2.000000e-01, 3.000000e-01}>>
}
// CHECK-NOT: onnx.MatMul
// CHECK-NOT: onnx.Add
// CHECK: "onnx.XFEMatMulBias"
