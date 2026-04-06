// RUN: onnx-mlir-opt --split-input-file --dq-binary-q-opt %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// Case 0: DQ -> BinaryOp(constant) -> Q   (existing full-sandwich pattern)
//
// Folds the constant into DQ's x_scale / x_zero_point when there is no
// removable Q->DQ chain before the DQ.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @dq_mul_q
// DQ(s=0.5, zp=0) -> Mul(k=2.0) -> Q(s=0.5, zp=0)
// Fold into DQ:  s' = 0.5 * 2.0 = 1.0
func.func @dq_mul_q(%arg0: tensor<1x4xui8>) -> tensor<1x4xui8> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %k = onnx.Constant dense<2.000000e+00> : tensor<1xf32>
  %q_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %q_zp = onnx.Constant dense<0> : tensor<ui8>

  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %mul = "onnx.Mul"(%dq, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%mul, %q_scale, %q_zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}
// DQ scale updated from 0.5 to 1.0
// CHECK-DAG: %[[NEW_S:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<0> : tensor<ui8>
// CHECK-DAG: %[[QS:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%arg0, %[[NEW_S]], %[[ZP]])
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%[[DQ]], %[[QS]],
// CHECK-NOT: onnx.Mul

// -----

// CHECK-LABEL: func.func @dq_add_q
// DQ(s=0.5, zp=10) -> Add(k=1.0) -> Q(s=0.5, zp=10)
// Fold into DQ:  zp' = 10 - 1.0/0.5 = 8
func.func @dq_add_q(%arg0: tensor<1x4xui8>) -> tensor<1x4xui8> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<10> : tensor<ui8>
  %k = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
  %q_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %q_zp = onnx.Constant dense<10> : tensor<ui8>

  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %add = "onnx.Add"(%dq, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%add, %q_scale, %q_zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}
// DQ zero-point updated from 10 to 8
// CHECK-DAG: %[[S:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG: %[[NEW_ZP:.*]] = onnx.Constant dense<8> : tensor<ui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%arg0, %[[S]], %[[NEW_ZP]])
// CHECK: "onnx.QuantizeLinear"(%[[DQ]],
// CHECK-NOT: onnx.Add

// -----

//===----------------------------------------------------------------------===//
// Case 1: activation -> BinaryOp(constant) -> Q   (no DQ before BinaryOp)
//
// Folds the constant into Q's y_scale / y_zero_point.
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

// CHECK-LABEL: func.func @div_q
// activation -> Div(k=2.0) -> Q(s=0.5, zp=0)
// Fold into Q:  s' = 0.5 * 2.0 = 1.0
func.func @div_q(%arg0: tensor<1x4xf32>) -> tensor<1x4xui8> {
  %k = onnx.Constant dense<2.000000e+00> : tensor<1xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>

  %div = "onnx.Div"(%arg0, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%div, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}
// Q scale updated from 0.5 to 1.0
// CHECK-DAG: %[[NEW_S:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<0> : tensor<ui8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[NEW_S]], %[[ZP]])
// CHECK-NOT: onnx.Div

// -----

// CHECK-LABEL: func.func @add_q
// activation -> Add(k=1.0) -> Q(s=0.5, zp=10)
// Fold into Q:  zp' = 10 + 1.0/0.5 = 12
func.func @add_q(%arg0: tensor<1x4xf32>) -> tensor<1x4xui8> {
  %k = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<10> : tensor<ui8>

  %add = "onnx.Add"(%arg0, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%add, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}
// Q zero-point updated from 10 to 12
// CHECK-DAG: %[[S:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG: %[[NEW_ZP:.*]] = onnx.Constant dense<12> : tensor<ui8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[NEW_ZP]])
// CHECK-NOT: onnx.Add

// -----

// CHECK-LABEL: func.func @sub_q
// activation -> Sub(k=1.0) -> Q(s=0.5, zp=10)
// Fold into Q:  zp' = 10 - 1.0/0.5 = 8
func.func @sub_q(%arg0: tensor<1x4xf32>) -> tensor<1x4xui8> {
  %k = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<10> : tensor<ui8>

  %sub = "onnx.Sub"(%arg0, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%sub, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}
// Q zero-point updated from 10 to 8
// CHECK-DAG: %[[S:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG: %[[NEW_ZP:.*]] = onnx.Constant dense<8> : tensor<ui8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[NEW_ZP]])
// CHECK-NOT: onnx.Sub

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
// Case 2: DQ -> BinaryOp(constant) -> consumer   (no Q after BinaryOp)
//
// Folds the constant into DQ's x_scale / x_zero_point.
// DQ must have a single use (the BinaryOp) for safe in-place modification.
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

// CHECK-LABEL: func.func @dq_div
// DQ(s=0.5, zp=0) -> Div(k=2.0) -> Relu
// Fold into DQ:  s' = 0.5 / 2.0 = 0.25
func.func @dq_div(%arg0: tensor<1x4xui8>) -> tensor<1x4xf32> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %k = onnx.Constant dense<2.000000e+00> : tensor<1xf32>

  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %div = "onnx.Div"(%dq, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %relu = "onnx.Relu"(%div) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %relu : tensor<1x4xf32>
}
// DQ scale updated from 0.5 to 0.25
// CHECK-DAG: %[[NEW_S:.*]] = onnx.Constant dense<2.500000e-01> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<0> : tensor<ui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%arg0, %[[NEW_S]], %[[ZP]])
// CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[DQ]])
// CHECK-NOT: onnx.Div

// -----

// CHECK-LABEL: func.func @dq_add
// DQ(s=0.5, zp=10) -> Add(k=1.0) -> Relu
// Fold into DQ:  zp' = 10 - 1.0/0.5 = 8
func.func @dq_add(%arg0: tensor<1x4xui8>) -> tensor<1x4xf32> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<10> : tensor<ui8>
  %k = onnx.Constant dense<1.000000e+00> : tensor<1xf32>

  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %add = "onnx.Add"(%dq, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %relu = "onnx.Relu"(%add) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %relu : tensor<1x4xf32>
}
// DQ zero-point updated from 10 to 8
// CHECK-DAG: %[[S:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG: %[[NEW_ZP:.*]] = onnx.Constant dense<8> : tensor<ui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%arg0, %[[S]], %[[NEW_ZP]])
// CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[DQ]])
// CHECK-NOT: onnx.Add

// -----

// CHECK-LABEL: func.func @dq_sub
// DQ(s=0.5, zp=10) -> Sub(k=1.0) -> Relu
// Fold into DQ:  zp' = 10 + 1.0/0.5 = 12
func.func @dq_sub(%arg0: tensor<1x4xui8>) -> tensor<1x4xf32> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<10> : tensor<ui8>
  %k = onnx.Constant dense<1.000000e+00> : tensor<1xf32>

  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %sub = "onnx.Sub"(%dq, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %relu = "onnx.Relu"(%sub) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %relu : tensor<1x4xf32>
}
// DQ zero-point updated from 10 to 12
// CHECK-DAG: %[[S:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG: %[[NEW_ZP:.*]] = onnx.Constant dense<12> : tensor<ui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%arg0, %[[S]], %[[NEW_ZP]])
// CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[DQ]])
// CHECK-NOT: onnx.Sub

// -----

//===----------------------------------------------------------------------===//
// Negative Tests - Should NOT be transformed
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @neg_sub_const_first_q
// Sub with constant as first operand -> Q: not supported (non-commutative)
func.func @neg_sub_const_first_q(%arg0: tensor<1x4xf32>) -> tensor<1x4xui8> {
  %k = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<10> : tensor<ui8>

  %sub = "onnx.Sub"(%k, %arg0) : (tensor<1xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%sub, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}
// CHECK: "onnx.Sub"
// CHECK: "onnx.QuantizeLinear"

// -----

// CHECK-LABEL: func.func @neg_div_const_first_dq
// Div with constant as first operand after DQ: not supported (non-commutative)
func.func @neg_div_const_first_dq(%arg0: tensor<1x4xui8>) -> tensor<1x4xf32> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %k = onnx.Constant dense<2.000000e+00> : tensor<1xf32>

  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %div = "onnx.Div"(%k, %dq) : (tensor<1xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %relu = "onnx.Relu"(%div) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %relu : tensor<1x4xf32>
}
// CHECK: "onnx.DequantizeLinear"
// CHECK: "onnx.Div"
// CHECK: "onnx.Relu"

// -----

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

// -----

// CHECK-LABEL: func.func @neg_div_k_zero_into_dq
// DQ -> Div(k=0): division by zero when computing s/k, no transform
func.func @neg_div_k_zero_into_dq(%arg0: tensor<1x4xui8>) -> tensor<1x4xf32> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %k = onnx.Constant dense<0.000000e+00> : tensor<1xf32>

  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %div = "onnx.Div"(%dq, %k) : (tensor<1x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  %relu = "onnx.Relu"(%div) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %relu : tensor<1x4xf32>
}
// CHECK: "onnx.DequantizeLinear"
// CHECK: "onnx.Div"
// CHECK: "onnx.Relu"
