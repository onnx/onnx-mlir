// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
/// Verification tests for AMD Quark Extended Quantize/Dequantize Operations
/// Domain: com.amd.quark
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// AMDQuarkExtendedQuantizeLinearOp: zero_point type must match result type
//===----------------------------------------------------------------------===//

// Test: y_zero_point element type (i16) does not match result element type (i8)
func.func @test_eq_zp_type_mismatch(%arg0: tensor<5x2x3x4xf32>, %arg1: tensor<f32>, %arg2: tensor<i16>) -> tensor<5x2x3x4xi8> {
  // expected-error @+1 {{'onnx.AMDQuarkExtendedQuantizeLinearOp' op y_zero_point element type 'i16' must match result element type 'i8'}}
  %0 = "onnx.AMDQuarkExtendedQuantizeLinearOp"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<5x2x3x4xf32>, tensor<f32>, tensor<i16>) -> tensor<5x2x3x4xi8>
  return %0 : tensor<5x2x3x4xi8>
}

// -----

// Test: y_zero_point element type (f16) does not match result element type (i8)
func.func @test_eq_zp_float_vs_int(%arg0: tensor<5x2x3x4xf32>, %arg1: tensor<f32>, %arg2: tensor<f16>) -> tensor<5x2x3x4xi8> {
  // expected-error @+1 {{'onnx.AMDQuarkExtendedQuantizeLinearOp' op y_zero_point element type 'f16' must match result element type 'i8'}}
  %0 = "onnx.AMDQuarkExtendedQuantizeLinearOp"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<5x2x3x4xf32>, tensor<f32>, tensor<f16>) -> tensor<5x2x3x4xi8>
  return %0 : tensor<5x2x3x4xi8>
}

// -----

// Test: none zero_point is allowed (no type check)
func.func @test_eq_none_zp_ok(%arg0: tensor<5x2x3x4xf32>, %arg1: tensor<f32>) -> tensor<5x2x3x4xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.AMDQuarkExtendedQuantizeLinearOp"(%arg0, %arg1, %none) {axis = 1 : si64} : (tensor<5x2x3x4xf32>, tensor<f32>, none) -> tensor<5x2x3x4xi8>
  return %0 : tensor<5x2x3x4xi8>
}

// -----

// Test: matching types should pass
func.func @test_eq_zp_type_match(%arg0: tensor<5x2x3x4xf32>, %arg1: tensor<f32>, %arg2: tensor<i8>) -> tensor<5x2x3x4xi8> {
  %0 = "onnx.AMDQuarkExtendedQuantizeLinearOp"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<5x2x3x4xf32>, tensor<f32>, tensor<i8>) -> tensor<5x2x3x4xi8>
  return %0 : tensor<5x2x3x4xi8>
}

// -----

//===----------------------------------------------------------------------===//
/// AMDQuarkExtendedDequantizeLinearOp: zero_point type must match input type
//===----------------------------------------------------------------------===//

// Test: x_zero_point element type (i16) does not match input element type (i8)
func.func @test_edq_zp_type_mismatch(%arg0: tensor<5x2x3x4xi8>, %arg1: tensor<f32>, %arg2: tensor<i16>) -> tensor<5x2x3x4xf32> {
  // expected-error @+1 {{'onnx.AMDQuarkExtendedDequantizeLinearOp' op x_zero_point element type 'i16' must match input element type 'i8'}}
  %0 = "onnx.AMDQuarkExtendedDequantizeLinearOp"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<5x2x3x4xi8>, tensor<f32>, tensor<i16>) -> tensor<5x2x3x4xf32>
  return %0 : tensor<5x2x3x4xf32>
}

// -----

// Test: x_zero_point element type (bf16) does not match input element type (f16)
func.func @test_edq_zp_bf16_vs_f16(%arg0: tensor<5x2x3x4xf16>, %arg1: tensor<f32>, %arg2: tensor<bf16>) -> tensor<5x2x3x4xf32> {
  // expected-error @+1 {{'onnx.AMDQuarkExtendedDequantizeLinearOp' op x_zero_point element type 'bf16' must match input element type 'f16'}}
  %0 = "onnx.AMDQuarkExtendedDequantizeLinearOp"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<5x2x3x4xf16>, tensor<f32>, tensor<bf16>) -> tensor<5x2x3x4xf32>
  return %0 : tensor<5x2x3x4xf32>
}

// -----

// Test: none zero_point is allowed (no type check)
func.func @test_edq_none_zp_ok(%arg0: tensor<5x2x3x4xi8>, %arg1: tensor<f32>) -> tensor<5x2x3x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.AMDQuarkExtendedDequantizeLinearOp"(%arg0, %arg1, %none) {axis = 1 : si64} : (tensor<5x2x3x4xi8>, tensor<f32>, none) -> tensor<5x2x3x4xf32>
  return %0 : tensor<5x2x3x4xf32>
}

// -----

// Test: matching types should pass
func.func @test_edq_zp_type_match(%arg0: tensor<5x2x3x4xi8>, %arg1: tensor<f32>, %arg2: tensor<i8>) -> tensor<5x2x3x4xf32> {
  %0 = "onnx.AMDQuarkExtendedDequantizeLinearOp"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<5x2x3x4xi8>, tensor<f32>, tensor<i8>) -> tensor<5x2x3x4xf32>
  return %0 : tensor<5x2x3x4xf32>
}
