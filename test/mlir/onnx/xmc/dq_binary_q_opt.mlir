// RUN: onnx-mlir-opt --split-input-file --dq-binary-q-opt %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// NOTE: Pass currently registers FoldBinaryIntoQ<ONNXMulOp> and
// FoldBinaryIntoDQ<ONNXMulOp> only. Other test sections are commented out below.

//===----------------------------------------------------------------------===//
// Case 1: activation -> Mul(constant) -> Q   (FoldBinaryIntoQ Mul)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mul_q
// activation -> Mul(k=2.0) -> Q(s=0.5, zp=0)
// Fold into Q:  s' = 0.5 / 2.0 = 0.25
func.func @mul_q(%arg0: tensor<1x4xf32>) -> tensor<1x4xui8> {
  %k = onnx.Constant dense<2.000000e+00> : tensor<1xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>

  %mul = "onnx.Mul"(%arg0, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%mul, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}
// Q scale updated from 0.5 to 0.25
// CHECK-DAG: %[[NEW_S:.*]] = onnx.Constant dense<2.500000e-01> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<0> : tensor<ui8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[NEW_S]], %[[ZP]])
// CHECK-NOT: onnx.Mul

// -----

// CHECK-LABEL: func.func @mul_q_const_first
// Mul is commutative: constant as first operand should also fold.
// Constant(k=2.0) * activation -> Q(s=0.5, zp=0)
// Fold into Q:  s' = 0.5 / 2.0 = 0.25
func.func @mul_q_const_first(%arg0: tensor<1x4xf32>) -> tensor<1x4xui8> {
  %k = onnx.Constant dense<2.000000e+00> : tensor<1xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>

  %mul = "onnx.Mul"(%k, %arg0) : (tensor<1xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%mul, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}
// CHECK-DAG: %[[NEW_S:.*]] = onnx.Constant dense<2.500000e-01> : tensor<f32>
// CHECK: "onnx.QuantizeLinear"(%arg0, %[[NEW_S]],
// CHECK-NOT: onnx.Mul

// -----

//===----------------------------------------------------------------------===//
// Case 2: DQ -> Mul(constant) -> consumer   (FoldBinaryIntoDQ Mul)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @dq_mul
// DQ(s=0.5, zp=0) -> Mul(k=2.0) -> Relu
// Fold into DQ:  s' = 0.5 * 2.0 = 1.0
func.func @dq_mul(%arg0: tensor<1x4xui8>) -> tensor<1x4xf32> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %k = onnx.Constant dense<2.000000e+00> : tensor<1xf32>

  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %mul = "onnx.Mul"(%dq, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %relu = "onnx.Relu"(%mul) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %relu : tensor<1x4xf32>
}
// DQ scale updated from 0.5 to 1.0
// CHECK-DAG: %[[NEW_S:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<0> : tensor<ui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%arg0, %[[NEW_S]], %[[ZP]])
// CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[DQ]])
// CHECK: return %[[RELU]]
// CHECK-NOT: onnx.Mul

// -----

//===----------------------------------------------------------------------===//
// Negative Tests - Mul patterns (should NOT be transformed)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @neg_mul_no_constant
// BinaryOp with two activations (no constant): no transform
func.func @neg_mul_no_constant(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xui8> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>

  %mul = "onnx.Mul"(%arg0, %arg1) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%mul, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}
// CHECK: "onnx.Mul"
// CHECK: "onnx.QuantizeLinear"

// -----

// CHECK-LABEL: func.func @neg_mul_multiple_uses
// BinaryOp with multiple uses: no transform
func.func @neg_mul_multiple_uses(%arg0: tensor<1x4xf32>) -> (tensor<1x4xui8>, tensor<1x4xf32>) {
  %k = onnx.Constant dense<2.000000e+00> : tensor<1xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>

  %mul = "onnx.Mul"(%arg0, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%mul, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q, %mul : tensor<1x4xui8>, tensor<1x4xf32>
}
// CHECK: "onnx.Mul"
// CHECK: "onnx.QuantizeLinear"

// -----

// CHECK-LABEL: func.func @neg_dq_mul_multi_use_dq
// DQ has multiple users: unsafe to modify in place, no transform for Case 2
func.func @neg_dq_mul_multi_use_dq(%arg0: tensor<1x4xui8>) -> (tensor<1x4xf32>, tensor<1x4xf32>) {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %k = onnx.Constant dense<2.000000e+00> : tensor<1xf32>

  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %mul = "onnx.Mul"(%dq, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  return %mul, %dq : tensor<1x4xf32>, tensor<1x4xf32>
}
// DQ has two users (mul and return), so FoldBinaryIntoDQ should not fire
// CHECK: "onnx.DequantizeLinear"
// CHECK: "onnx.Mul"

// -----

// CHECK-LABEL: func.func @neg_mul_k_zero_into_q
// Mul with k=0 -> Q: division by zero when computing s/k, no transform
func.func @neg_mul_k_zero_into_q(%arg0: tensor<1x4xf32>) -> tensor<1x4xui8> {
  %k = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>

  %mul = "onnx.Mul"(%arg0, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%mul, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}
// CHECK: "onnx.Mul"
// CHECK: "onnx.QuantizeLinear"

// DISABLED: tests below require patterns not currently registered in
// DQBinaryQOptPass (FoldBinaryThroughQDQ, FoldBinaryIntoQ/DQ for Add/Sub/Div).
//
// Case 0: @dq_mul_q, @dq_add_q
// Case 1: @div_q, @add_q, @sub_q
// Case 2: @dq_div, @dq_add, @dq_sub
// Negative: @neg_sub_const_first_q, @neg_div_const_first_dq, @neg_div_k_zero_into_dq
