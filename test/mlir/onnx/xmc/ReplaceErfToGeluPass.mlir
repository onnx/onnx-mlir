// RUN: onnx-mlir-opt --split-input-file --replace-erf-to-gelu %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// ============================================================================
// PASS CASES: Erf-based GELU with quantized types
// ============================================================================

// CHECK-LABEL: @erf_gelu_quantized_u8_pass
func.func @erf_gelu_quantized_u8_pass(%arg0: tensor<1x64x128x128x!quant.uniform<u8:f32, 0.031890511512756348:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.030503427609801292:128>> {
  %sqrt2 = onnx.Constant {value = dense<128> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.011091871187090874:128>>
  %one = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0078431377187371254:128>>
  %half = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0039215688593685627:128>>
  %div = "onnx.Div"(%arg0, %sqrt2) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.031890511512756348:128>>, tensor<!quant.uniform<u8:f32, 0.011091871187090874:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.02254999615252018:128>>
  %erf = "onnx.Erf"(%div) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.02254999615252018:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.0078427623957395554:128>>
  %add = "onnx.Add"(%erf, %one) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.0078427623957395554:128>>, tensor<!quant.uniform<u8:f32, 0.0078431377187371254:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.015685485675930977:128>>
  %mul = "onnx.Mul"(%arg0, %add) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.031890511512756348:128>>, tensor<1x64x128x128x!quant.uniform<u8:f32, 0.015685485675930977:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.061006855219602585:128>>
  %mul1 = "onnx.Mul"(%mul, %half) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.061006855219602585:128>>, tensor<!quant.uniform<u8:f32, 0.0039215688593685627:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.030503427609801292:128>>
  return %mul1 : tensor<1x64x128x128x!quant.uniform<u8:f32, 0.030503427609801292:128>>
}
// CHECK-NOT: "onnx.Erf"
// CHECK-NOT: "onnx.Div"
// CHECK: %[[GELU:.*]] = "onnx.Gelu"(%arg0) {approximate = "none"}
// CHECK-SAME: (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.031890511512756348:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.030503427609801292:128>>
// CHECK: return %[[GELU]]

// -----

// Test with reversed operand order in Mul (add_result * x instead of x * add_result)
// CHECK-LABEL: @erf_gelu_reversed_mul_operands_pass
func.func @erf_gelu_reversed_mul_operands_pass(%arg0: tensor<1x32x128x128x!quant.uniform<u8:f32, 0.05:128>>) -> tensor<1x32x128x128x!quant.uniform<u8:f32, 0.02:128>> {
  %sqrt2 = onnx.Constant {value = dense<128> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.011:128>>
  %one = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0078:128>>
  %half = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.004:128>>
  %div = "onnx.Div"(%arg0, %sqrt2) : (tensor<1x32x128x128x!quant.uniform<u8:f32, 0.05:128>>, tensor<!quant.uniform<u8:f32, 0.011:128>>) -> tensor<1x32x128x128x!quant.uniform<u8:f32, 0.035:128>>
  %erf = "onnx.Erf"(%div) : (tensor<1x32x128x128x!quant.uniform<u8:f32, 0.035:128>>) -> tensor<1x32x128x128x!quant.uniform<u8:f32, 0.008:128>>
  %add = "onnx.Add"(%erf, %one) : (tensor<1x32x128x128x!quant.uniform<u8:f32, 0.008:128>>, tensor<!quant.uniform<u8:f32, 0.0078:128>>) -> tensor<1x32x128x128x!quant.uniform<u8:f32, 0.016:128>>
  %mul = "onnx.Mul"(%add, %arg0) : (tensor<1x32x128x128x!quant.uniform<u8:f32, 0.016:128>>, tensor<1x32x128x128x!quant.uniform<u8:f32, 0.05:128>>) -> tensor<1x32x128x128x!quant.uniform<u8:f32, 0.04:128>>
  %mul1 = "onnx.Mul"(%mul, %half) : (tensor<1x32x128x128x!quant.uniform<u8:f32, 0.04:128>>, tensor<!quant.uniform<u8:f32, 0.004:128>>) -> tensor<1x32x128x128x!quant.uniform<u8:f32, 0.02:128>>
  return %mul1 : tensor<1x32x128x128x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK-NOT: "onnx.Erf"
// CHECK: "onnx.Gelu"(%arg0) {approximate = "none"}
// CHECK-SAME: -> tensor<1x32x128x128x!quant.uniform<u8:f32, 2.000000e-02:128>>

// -----

// Test with i8 quantized types
// CHECK-LABEL: @erf_gelu_quantized_i8_pass
func.func @erf_gelu_quantized_i8_pass(%arg0: tensor<1x64x16x16x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>> {
  %sqrt2 = onnx.Constant {value = dense<0> : tensor<i8>} : tensor<!quant.uniform<i8:f32, 0.011:0>>
  %one = onnx.Constant {value = dense<127> : tensor<i8>} : tensor<!quant.uniform<i8:f32, 0.008:0>>
  %half = onnx.Constant {value = dense<64> : tensor<i8>} : tensor<!quant.uniform<i8:f32, 0.004:0>>
  %div = "onnx.Div"(%arg0, %sqrt2) : (tensor<1x64x16x16x!quant.uniform<i8:f32, 0.1:0>>, tensor<!quant.uniform<i8:f32, 0.011:0>>) -> tensor<1x64x16x16x!quant.uniform<i8:f32, 0.07:0>>
  %erf = "onnx.Erf"(%div) : (tensor<1x64x16x16x!quant.uniform<i8:f32, 0.07:0>>) -> tensor<1x64x16x16x!quant.uniform<i8:f32, 0.008:0>>
  %add = "onnx.Add"(%erf, %one) : (tensor<1x64x16x16x!quant.uniform<i8:f32, 0.008:0>>, tensor<!quant.uniform<i8:f32, 0.008:0>>) -> tensor<1x64x16x16x!quant.uniform<i8:f32, 0.016:0>>
  %mul = "onnx.Mul"(%arg0, %add) : (tensor<1x64x16x16x!quant.uniform<i8:f32, 0.1:0>>, tensor<1x64x16x16x!quant.uniform<i8:f32, 0.016:0>>) -> tensor<1x64x16x16x!quant.uniform<i8:f32, 0.1:0>>
  %mul1 = "onnx.Mul"(%mul, %half) : (tensor<1x64x16x16x!quant.uniform<i8:f32, 0.1:0>>, tensor<!quant.uniform<i8:f32, 0.004:0>>) -> tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>>
  return %mul1 : tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK-NOT: "onnx.Erf"
// CHECK: "onnx.Gelu"(%arg0) {approximate = "none"}
// CHECK-SAME: -> tensor<1x64x16x16x!quant.uniform<i8:f32, 5.000000e-02>>

// ============================================================================
// FAIL CASES: Patterns that should NOT be transformed
// ============================================================================

// -----

// Non-quantized types should not be transformed
// CHECK-LABEL: @erf_gelu_non_quantized_fail
func.func @erf_gelu_non_quantized_fail(%arg0: tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32> {
  %sqrt2 = onnx.Constant dense<1.41421356> : tensor<f32>
  %one = onnx.Constant dense<1.0> : tensor<f32>
  %half = onnx.Constant dense<0.5> : tensor<f32>
  %div = "onnx.Div"(%arg0, %sqrt2) : (tensor<1x64x128x128xf32>, tensor<f32>) -> tensor<1x64x128x128xf32>
  %erf = "onnx.Erf"(%div) : (tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
  %add = "onnx.Add"(%erf, %one) : (tensor<1x64x128x128xf32>, tensor<f32>) -> tensor<1x64x128x128xf32>
  %mul = "onnx.Mul"(%arg0, %add) : (tensor<1x64x128x128xf32>, tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
  %mul1 = "onnx.Mul"(%mul, %half) : (tensor<1x64x128x128xf32>, tensor<f32>) -> tensor<1x64x128x128xf32>
  return %mul1 : tensor<1x64x128x128xf32>
}
// CHECK: "onnx.Erf"
// CHECK-NOT: "onnx.Gelu"

// -----

// Erf input is not Div — should not match
// CHECK-LABEL: @erf_gelu_no_div_fail
func.func @erf_gelu_no_div_fail(%arg0: tensor<1x64x128x128x!quant.uniform<u8:f32, 0.03:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.02:128>> {
  %one = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0078:128>>
  %half = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.004:128>>
  %erf = "onnx.Erf"(%arg0) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.03:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.008:128>>
  %add = "onnx.Add"(%erf, %one) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.008:128>>, tensor<!quant.uniform<u8:f32, 0.0078:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.016:128>>
  %mul = "onnx.Mul"(%arg0, %add) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.03:128>>, tensor<1x64x128x128x!quant.uniform<u8:f32, 0.016:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.04:128>>
  %mul1 = "onnx.Mul"(%mul, %half) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.04:128>>, tensor<!quant.uniform<u8:f32, 0.004:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.02:128>>
  return %mul1 : tensor<1x64x128x128x!quant.uniform<u8:f32, 0.02:128>>
}
// CHECK: "onnx.Erf"
// CHECK-NOT: "onnx.Gelu"

// -----

// Erf has multiple uses — should not match
// CHECK-LABEL: @erf_gelu_multi_use_fail
func.func @erf_gelu_multi_use_fail(%arg0: tensor<1x64x128x128x!quant.uniform<u8:f32, 0.05:128>>) -> (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.02:128>>, tensor<1x64x128x128x!quant.uniform<u8:f32, 0.008:128>>) {
  %sqrt2 = onnx.Constant {value = dense<128> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.011:128>>
  %one = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0078:128>>
  %half = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.004:128>>
  %div = "onnx.Div"(%arg0, %sqrt2) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.05:128>>, tensor<!quant.uniform<u8:f32, 0.011:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.035:128>>
  %erf = "onnx.Erf"(%div) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.035:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.008:128>>
  %add = "onnx.Add"(%erf, %one) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.008:128>>, tensor<!quant.uniform<u8:f32, 0.0078:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.016:128>>
  %mul = "onnx.Mul"(%arg0, %add) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.05:128>>, tensor<1x64x128x128x!quant.uniform<u8:f32, 0.016:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.04:128>>
  %mul1 = "onnx.Mul"(%mul, %half) : (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.04:128>>, tensor<!quant.uniform<u8:f32, 0.004:128>>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.02:128>>
  return %mul1, %erf : tensor<1x64x128x128x!quant.uniform<u8:f32, 0.02:128>>, tensor<1x64x128x128x!quant.uniform<u8:f32, 0.008:128>>
}
// CHECK: "onnx.Erf"
// CHECK-NOT: "onnx.Gelu"
