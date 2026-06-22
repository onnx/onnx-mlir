// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

// Verify that canonicalization does not crash when an onnx.Mul has
// an unranked result type (e.g., downstream of a Resize with empty
// optional NoneType inputs that prevented shape inference).
// Previously, FuseScaleIntoRotaryEmbeddingPattern called
// isPlainFloatType on a null RankedTensorType from dyn_cast, which
// called getElementTypeOrSelf on a null Type and hit:
//   "dyn_cast on a non-existent value"
//
// The scale operand MUST be a dense constant so the pattern advances
// past its first guard (isDenseONNXConstant) and reaches the result
// type check that triggers the crash.

// CHECK-LABEL: @mul_unranked_result_const_scale
func.func @mul_unranked_result_const_scale(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %scale = onnx.Constant dense<2.0> : tensor<1xf32>
  // Mul result is unranked (tensor<*xf32>).
  // FuseScaleIntoRotaryEmbeddingPattern matches ONNXMulOp, passes the
  // isDenseONNXConstant / scaleTy / isPlainFloatType(scaleTy) checks,
  // then hits isPlainFloatType on the unranked result type.
  // Without the fix this crashes; with the fix the pattern gracefully
  // fails to match and the Mul is preserved.
  %0 = "onnx.Mul"(%arg0, %scale) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
  // CHECK: "onnx.Mul"
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Same scenario but with a Resize producing the unranked value that
// feeds into a Mul with a constant scale.

// CHECK-LABEL: @resize_none_inputs_mul_const_scale
func.func @resize_none_inputs_mul_const_scale(%data: tensor<1x1x2x2xf32>, %sizes: tensor<4xi64>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scale = onnx.Constant dense<0.5> : tensor<1xf32>
  %resized = "onnx.Resize"(%data, %none, %none, %sizes) {
    antialias = 0 : si64,
    coordinate_transformation_mode = "half_pixel",
    cubic_coeff_a = -7.500000e-01 : f32,
    exclude_outside = 0 : si64,
    extrapolation_value = 0.000000e+00 : f32,
    keep_aspect_ratio_policy = "stretch",
    mode = "nearest",
    nearest_mode = "floor"
  } : (tensor<1x1x2x2xf32>, none, none, tensor<4xi64>) -> tensor<*xf32>
  %result = "onnx.Mul"(%resized, %scale) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
  // CHECK: "onnx.Mul"
  onnx.Return %result : tensor<*xf32>
}
