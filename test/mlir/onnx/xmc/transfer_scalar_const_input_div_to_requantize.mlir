// RUN: onnx-mlir-opt --transfer-scalar-const-input-div-to-requantize %s --split-input-file | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// Positive Tests: Should convert Div / Mul (with scalar UI16 const RHS) to
// XCOMPILERRequantize. All scales are chosen to be exactly representable in
// f32 so the rewrite is bit-exact. The Div / Mul result is consumed by an
// ONNX Relu so the rewrite is allowed to retype the result.
//===----------------------------------------------------------------------===//

// Test 1: Div by a scalar UI16 quantized constant, real_c = 1.0.
// real_c = (2 - 0) * 0.5 = 1.0; new_y_scale = 0.25 * 1.0 = 0.25 (unchanged).
// CHECK-LABEL: @div_scalar_ui16_const_unit
func.func @div_scalar_ui16_const_unit(%arg0: tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>> {
  %c = onnx.Constant {value = dense<2> : tensor<1xui16>} : tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>
  %0 = "onnx.Div"(%arg0, %c) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>, tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %1 = "onnx.Relu"(%0) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>>
  return %1 : tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>>
}
// CHECK-NOT: onnx.Div
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: a_scale = [5.000000e-01
// CHECK-SAME: a_zero_point = [0]
// CHECK-SAME: y_scale = [2.500000e-01
// CHECK-SAME: y_zero_point = [0]

// -----

// Test 2: Div by a scalar UI16 quantized constant, real_c = 0.5.
// real_c = (2 - 0) * 0.25 = 0.5; new_y_scale = 0.25 * 0.5 = 0.125.
// CHECK-LABEL: @div_scalar_ui16_const_half
func.func @div_scalar_ui16_const_half(%arg0: tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 6.250000e-02:0>> {
  %c = onnx.Constant {value = dense<2> : tensor<1xui16>} : tensor<1x!quant.uniform<u16:f32, 2.500000e-01:0>>
  %0 = "onnx.Div"(%arg0, %c) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>, tensor<1x!quant.uniform<u16:f32, 2.500000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %1 = "onnx.Relu"(%0) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 6.250000e-02:0>>
  return %1 : tensor<1x32x7x7x!quant.uniform<u8:f32, 6.250000e-02:0>>
}
// CHECK-NOT: onnx.Div
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: a_scale = [5.000000e-01
// CHECK-SAME: y_scale = [1.250000e-01

// -----

// Test 3: Mul by a scalar UI16 quantized constant, real_c = 2.0.
// real_c = (4 - 0) * 0.5 = 2.0; new_y_scale = 0.5 / 2.0 = 0.25.
// CHECK-LABEL: @mul_scalar_ui16_const
func.func @mul_scalar_ui16_const(%arg0: tensor<1x16x14x14x!quant.uniform<u8:f32, 2.500000e-01:0>>) -> tensor<1x16x14x14x!quant.uniform<u8:f32, 1.250000e-01:0>> {
  %c = onnx.Constant {value = dense<4> : tensor<1xui16>} : tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>
  %0 = "onnx.Mul"(%arg0, %c) : (tensor<1x16x14x14x!quant.uniform<u8:f32, 2.500000e-01:0>>, tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>) -> tensor<1x16x14x14x!quant.uniform<u8:f32, 5.000000e-01:0>>
  %1 = "onnx.Relu"(%0) : (tensor<1x16x14x14x!quant.uniform<u8:f32, 5.000000e-01:0>>) -> tensor<1x16x14x14x!quant.uniform<u8:f32, 1.250000e-01:0>>
  return %1 : tensor<1x16x14x14x!quant.uniform<u8:f32, 1.250000e-01:0>>
}
// CHECK-NOT: onnx.Mul
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: a_scale = [2.500000e-01
// CHECK-SAME: y_scale = [2.500000e-01

// -----

//===----------------------------------------------------------------------===//
// Negative Tests: Should NOT convert.
//===----------------------------------------------------------------------===//

// Test 4: RHS constant is not UI16 (use I8) - storage type guard.
// CHECK-LABEL: @div_scalar_i8_const_no_match
func.func @div_scalar_i8_const_no_match(%arg0: tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>> {
  %c = onnx.Constant {value = dense<2> : tensor<1xi8>} : tensor<1x!quant.uniform<i8:f32, 5.000000e-01:0>>
  %0 = "onnx.Div"(%arg0, %c) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>, tensor<1x!quant.uniform<i8:f32, 5.000000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %1 = "onnx.Relu"(%0) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>>
  return %1 : tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>>
}
// CHECK: onnx.Div
// CHECK-NOT: onnx.XCOMPILERRequantize

// -----

// Test 5: RHS constant is not scalar (4 elements) - scalar-only filter.
// CHECK-LABEL: @div_vector_ui16_const_no_match
func.func @div_vector_ui16_const_no_match(%arg0: tensor<1x4x!quant.uniform<u8:f32, 5.000000e-01:0>>) -> tensor<1x4x!quant.uniform<u8:f32, 1.250000e-01:0>> {
  %c = onnx.Constant {value = dense<[2, 3, 4, 5]> : tensor<4xui16>} : tensor<4x!quant.uniform<u16:f32, 5.000000e-01:0>>
  %0 = "onnx.Div"(%arg0, %c) : (tensor<1x4x!quant.uniform<u8:f32, 5.000000e-01:0>>, tensor<4x!quant.uniform<u16:f32, 5.000000e-01:0>>) -> tensor<1x4x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %1 = "onnx.Relu"(%0) : (tensor<1x4x!quant.uniform<u8:f32, 2.500000e-01:0>>) -> tensor<1x4x!quant.uniform<u8:f32, 1.250000e-01:0>>
  return %1 : tensor<1x4x!quant.uniform<u8:f32, 1.250000e-01:0>>
}
// CHECK: onnx.Div
// CHECK-NOT: onnx.XCOMPILERRequantize

// -----

// Test 6: LHS is a constant - constant must be on RHS.
// CHECK-LABEL: @div_const_lhs_no_match
func.func @div_const_lhs_no_match(%arg0: tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>) -> tensor<1x!quant.uniform<u8:f32, 1.250000e-01:0>> {
  %c = onnx.Constant {value = dense<2> : tensor<1xui16>} : tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>
  %0 = "onnx.Div"(%c, %arg0) : (tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>, tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>) -> tensor<1x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %1 = "onnx.Relu"(%0) : (tensor<1x!quant.uniform<u8:f32, 2.500000e-01:0>>) -> tensor<1x!quant.uniform<u8:f32, 1.250000e-01:0>>
  return %1 : tensor<1x!quant.uniform<u8:f32, 1.250000e-01:0>>
}
// CHECK: onnx.Div
// CHECK-NOT: onnx.XCOMPILERRequantize

// -----

// Test 7: Dequantized constant is zero (data == zp) - divide-by-zero guard.
// CHECK-LABEL: @div_zero_dq_const_no_match
func.func @div_zero_dq_const_no_match(%arg0: tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>> {
  %c = onnx.Constant {value = dense<1> : tensor<1xui16>} : tensor<1x!quant.uniform<u16:f32, 5.000000e-01:1>>
  %0 = "onnx.Div"(%arg0, %c) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>, tensor<1x!quant.uniform<u16:f32, 5.000000e-01:1>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %1 = "onnx.Relu"(%0) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>>
  return %1 : tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>>
}
// CHECK: onnx.Div
// CHECK-NOT: onnx.XCOMPILERRequantize

// -----

// Test 8: Add is not supported - only Div / Mul match this pass.
// CHECK-LABEL: @add_scalar_ui16_const_no_match
func.func @add_scalar_ui16_const_no_match(%arg0: tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>> {
  %c = onnx.Constant {value = dense<2> : tensor<1xui16>} : tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>
  %0 = "onnx.Add"(%arg0, %c) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>, tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %1 = "onnx.Relu"(%0) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>>
  return %1 : tensor<1x32x7x7x!quant.uniform<u8:f32, 1.250000e-01:0>>
}
// CHECK: onnx.Add
// CHECK-NOT: onnx.XCOMPILERRequantize

// -----

// Test 9: Div consumed directly by func.return - still rewrites. The new
// pass keeps the result tensor's advertised quant type identical to the
// original Div output and only embeds the kernel scale in y_scale, so the
// type signature (and therefore the func.return contract) is preserved.
// real_c = (2 - 0) * 0.5 = 1.0; kernel y_scale = 0.25 * 1.0 = 0.25.
// CHECK-LABEL: @div_consumed_by_return
func.func @div_consumed_by_return(%arg0: tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>> {
  %c = onnx.Constant {value = dense<2> : tensor<1xui16>} : tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>
  %0 = "onnx.Div"(%arg0, %c) : (tensor<1x32x7x7x!quant.uniform<u8:f32, 5.000000e-01:0>>, tensor<1x!quant.uniform<u16:f32, 5.000000e-01:0>>) -> tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>
  return %0 : tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01:0>>
}
// CHECK-NOT: onnx.Div
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: a_scale = [5.000000e-01
// CHECK-SAME: y_scale = [2.500000e-01
// CHECK-SAME: -> tensor<1x32x7x7x!quant.uniform<u8:f32, 2.500000e-01
