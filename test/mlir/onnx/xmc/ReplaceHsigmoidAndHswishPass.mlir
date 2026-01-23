// RUN: onnx-mlir-opt --split-input-file --replace-hsigmoid-and-hswish %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// ============================================================================
// PASS CASES: HardSigmoid with quantized input and output types
// ============================================================================

// CHECK-LABEL: @hardsigmoid_quantized_u8_pass
func.func @hardsigmoid_quantized_u8_pass(%arg0: tensor<1x3x224x224x!quant.uniform<u8:f32, 0.05000000074505806:128>>) -> tensor<1x3x224x224x!quant.uniform<u8:f32, 0.0039215697906911373:0>> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 0.5 : f32} : (tensor<1x3x224x224x!quant.uniform<u8:f32, 0.05000000074505806:128>>) -> tensor<1x3x224x224x!quant.uniform<u8:f32, 0.0039215697906911373:0>>
  return %0 : tensor<1x3x224x224x!quant.uniform<u8:f32, 0.0039215697906911373:0>>
}
// CHECK-NOT: "onnx.HardSigmoid"
// CHECK: %[[NONE:.*]] = "onnx.NoValue"()
// CHECK: %[[ELTWISE:.*]] = "onnx.XCOMPILERFusedEltwise"
// CHECK-SAME: nonlinear = "NONE"
// CHECK-SAME: type = "QLINEARSIGMOID"
// CHECK-SAME: (tensor<1x3x224x224x!quant.uniform<u8:f32, 0.05000000074505806:128>>, none) -> tensor<1x3x224x224x!quant.uniform<u8:f32, 0.0039215697906911373>>
// -----

// CHECK-LABEL: @hardsigmoid_quantized_i8_pass
func.func @hardsigmoid_quantized_i8_pass(%arg0: tensor<1x16x28x28x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x16x28x28x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 0.5 : f32} : (tensor<1x16x28x28x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x16x28x28x!quant.uniform<i8:f32, 0.05:0>>
  return %0 : tensor<1x16x28x28x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK-NOT: "onnx.HardSigmoid"
// CHECK: %[[NONE:.*]] = "onnx.NoValue"()
// CHECK: %[[ELTWISE:.*]] = "onnx.XCOMPILERFusedEltwise"
// CHECK-SAME: type = "QLINEARSIGMOID"
// CHECK-SAME: (tensor<1x16x28x28x!quant.uniform<i8:f32, 1.000000e-01>>, none) -> tensor<1x16x28x28x!quant.uniform<i8:f32, 5.000000e-02>>

// -----

// CHECK-LABEL: @hardsigmoid_quantized_same_scale_zp_pass
func.func @hardsigmoid_quantized_same_scale_zp_pass(%arg0: tensor<2x3x4x5x!quant.uniform<u8:f32, 0.2:1>>) -> tensor<2x3x4x5x!quant.uniform<u8:f32, 0.2:1>> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 0.5 : f32} : (tensor<2x3x4x5x!quant.uniform<u8:f32, 0.2:1>>) -> tensor<2x3x4x5x!quant.uniform<u8:f32, 0.2:1>>
  return %0 : tensor<2x3x4x5x!quant.uniform<u8:f32, 0.2:1>>
}
// CHECK-NOT: "onnx.HardSigmoid"
// CHECK: "onnx.XCOMPILERFusedEltwise"
// CHECK-SAME: type = "QLINEARSIGMOID"

// ============================================================================
// FAIL CASES: HardSigmoid that should NOT be transformed
// ============================================================================

// CHECK-LABEL: @hardsigmoid_non_quantized_input_fail
func.func @hardsigmoid_non_quantized_input_fail(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<u8:f32, 0.0039215697906911373:0>> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 0.5 : f32} : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<u8:f32, 0.0039215697906911373:0>>
  return %0 : tensor<1x3x224x224x!quant.uniform<u8:f32, 0.0039215697906911373:0>>
}
// CHECK: "onnx.HardSigmoid"
// CHECK-NOT: "onnx.XCOMPILERFusedEltwise"

// -----

// CHECK-LABEL: @hardsigmoid_non_quantized_output_fail
func.func @hardsigmoid_non_quantized_output_fail(%arg0: tensor<1x3x224x224x!quant.uniform<u8:f32, 0.05000000074505806:128>>) -> tensor<1x3x224x224xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 0.5 : f32} : (tensor<1x3x224x224x!quant.uniform<u8:f32, 0.05000000074505806:128>>) -> tensor<1x3x224x224xf32>
  return %0 : tensor<1x3x224x224xf32>
}
// CHECK: "onnx.HardSigmoid"
// CHECK-NOT: "onnx.XCOMPILERFusedEltwise"

// -----

// CHECK-LABEL: @hardsigmoid_bf16_fail
func.func @hardsigmoid_bf16_fail(%arg0: tensor<1x3x224x224xbf16>) -> tensor<1x3x224x224xbf16> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 0.5 : f32} : (tensor<1x3x224x224xbf16>) -> tensor<1x3x224x224xbf16>
  return %0 : tensor<1x3x224x224xbf16>
}
// CHECK: "onnx.HardSigmoid"
// CHECK-NOT: "onnx.XCOMPILERFusedEltwise"

// -----

// CHECK-LABEL: @hardsigmoid_f16_fail
func.func @hardsigmoid_f16_fail(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 0.5 : f32} : (tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16>
  return %0 : tensor<1x3x224x224xf16>
}
// CHECK: "onnx.HardSigmoid"
// CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
