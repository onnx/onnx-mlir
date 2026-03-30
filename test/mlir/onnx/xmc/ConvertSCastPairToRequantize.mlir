// RUN: onnx-mlir-opt --convert-scast-pair-to-requantize %s --split-input-file | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// Positive Tests: Should convert to XCOMPILERRequantize
//===----------------------------------------------------------------------===//

// Test 1: Basic per-tensor requantization (different scales)
// CHECK-LABEL: @scast_pair_different_scale
func.func @scast_pair_different_scale(%arg0: tensor<1x32x7x7x!quant.uniform<i8:f32, 0.0084162093698978424:0>>) -> tensor<1x32x7x7x!quant.uniform<i8:f32, 0.0084700193256139755:0>> {
    %0 = quant.scast %arg0 : tensor<1x32x7x7x!quant.uniform<i8:f32, 0.0084162093698978424:0>> to tensor<1x32x7x7xi8>
    %1 = quant.scast %0 : tensor<1x32x7x7xi8> to tensor<1x32x7x7x!quant.uniform<i8:f32, 0.0084700193256139755:0>>
    return %1 : tensor<1x32x7x7x!quant.uniform<i8:f32, 0.0084700193256139755:0>>
}
// CHECK-NOT: quant.scast
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: a_scale
// CHECK-SAME: y_scale

// -----

// Test 2: Per-tensor requantization with different zero points
// CHECK-LABEL: @scast_pair_different_zp
func.func @scast_pair_different_zp(%arg0: tensor<1x64x14x14x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x64x14x14x!quant.uniform<i8:f32, 0.05:10>> {
    %0 = quant.scast %arg0 : tensor<1x64x14x14x!quant.uniform<i8:f32, 0.05:0>> to tensor<1x64x14x14xi8>
    %1 = quant.scast %0 : tensor<1x64x14x14xi8> to tensor<1x64x14x14x!quant.uniform<i8:f32, 0.05:10>>
    return %1 : tensor<1x64x14x14x!quant.uniform<i8:f32, 0.05:10>>
}
// CHECK-NOT: quant.scast
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: a_zero_point = [0]
// CHECK-SAME: y_zero_point = [10]

// -----

// Test 3: Both scale and zero point differ
// CHECK-LABEL: @scast_pair_different_scale_and_zp
func.func @scast_pair_different_scale_and_zp(%arg0: tensor<1x16x28x28x!quant.uniform<i8:f32, 0.03:-5>>) -> tensor<1x16x28x28x!quant.uniform<i8:f32, 0.07:3>> {
    %0 = quant.scast %arg0 : tensor<1x16x28x28x!quant.uniform<i8:f32, 0.03:-5>> to tensor<1x16x28x28xi8>
    %1 = quant.scast %0 : tensor<1x16x28x28xi8> to tensor<1x16x28x28x!quant.uniform<i8:f32, 0.07:3>>
    return %1 : tensor<1x16x28x28x!quant.uniform<i8:f32, 0.07:3>>
}
// CHECK-NOT: quant.scast
// CHECK: "onnx.XCOMPILERRequantize"

// -----

// Test 4: Signed i8 quantization
// CHECK-LABEL: @scast_pair_signed_i8
func.func @scast_pair_signed_i8(%arg0: tensor<1x3x224x224x!quant.uniform<i8:f32, 0.01:0>>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 0.02:0>> {
    %0 = quant.scast %arg0 : tensor<1x3x224x224x!quant.uniform<i8:f32, 0.01:0>> to tensor<1x3x224x224xi8>
    %1 = quant.scast %0 : tensor<1x3x224x224xi8> to tensor<1x3x224x224x!quant.uniform<i8:f32, 0.02:0>>
    return %1 : tensor<1x3x224x224x!quant.uniform<i8:f32, 0.02:0>>
}
// CHECK-NOT: quant.scast
// CHECK: "onnx.XCOMPILERRequantize"

// -----

//===----------------------------------------------------------------------===//
// Negative Tests: Should NOT convert
//===----------------------------------------------------------------------===//

// Test 5: Same quantization parameters - no requantization needed
// The scast pair with identical params is a no-op round-trip and gets folded away
// CHECK-LABEL: @scast_pair_same_params
func.func @scast_pair_same_params(%arg0: tensor<1x32x7x7x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x32x7x7x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = quant.scast %arg0 : tensor<1x32x7x7x!quant.uniform<i8:f32, 0.05:0>> to tensor<1x32x7x7xi8>
    %1 = quant.scast %0 : tensor<1x32x7x7xi8> to tensor<1x32x7x7x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x32x7x7x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK-NOT: onnx.XCOMPILERRequantize
// CHECK: return %arg0

// -----

// Test 6: Single scast (no pair) should not be modified
// CHECK-LABEL: @single_scast_no_pair
func.func @single_scast_no_pair(%arg0: tensor<1x32x7x7x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x32x7x7xi8> {
    %0 = quant.scast %arg0 : tensor<1x32x7x7x!quant.uniform<i8:f32, 0.05:0>> to tensor<1x32x7x7xi8>
    return %0 : tensor<1x32x7x7xi8>
}
// CHECK-NOT: onnx.XCOMPILERRequantize
// CHECK: quant.scast

// -----

// Test 7: scast pair where intermediate has multiple uses - should still convert
// the second scast to XCOMPILERRequantize; first scast stays for the other use.
// CHECK-LABEL: @scast_pair_multi_use
func.func @scast_pair_multi_use(%arg0: tensor<1x32x7x7x!quant.uniform<i8:f32, 0.05:0>>) -> (tensor<1x32x7x7x!quant.uniform<i8:f32, 0.07:0>>, tensor<1x32x7x7xi8>) {
    %0 = quant.scast %arg0 : tensor<1x32x7x7x!quant.uniform<i8:f32, 0.05:0>> to tensor<1x32x7x7xi8>
    %1 = quant.scast %0 : tensor<1x32x7x7xi8> to tensor<1x32x7x7x!quant.uniform<i8:f32, 0.07:0>>
    return %1, %0 : tensor<1x32x7x7x!quant.uniform<i8:f32, 0.07:0>>, tensor<1x32x7x7xi8>
}
// CHECK: quant.scast
// CHECK: "onnx.XCOMPILERRequantize"

// -----

//===----------------------------------------------------------------------===//
// Pattern 2: Q node -> Scast node (diff scale/zp) -> Requantize
//===----------------------------------------------------------------------===//

// Test 8: QuantizeLinear(scale_q,zp_q) -> quant.scast(to scale_s,zp_s) -> Requantize
// CHECK-LABEL: @q_then_scast_different_params
func.func @q_then_scast_different_params(%arg0: tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<u8:f32, 0.2:5>> {
  %scale_q = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %zp_q = onnx.Constant dense<0> : tensor<ui8>
  %0 = "onnx.QuantizeLinear"(%arg0, %scale_q, %zp_q) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x4xui8>
  %1 = quant.scast %0 : tensor<2x4xui8> to tensor<2x4x!quant.uniform<u8:f32, 0.2:5>>
  return %1 : tensor<2x4x!quant.uniform<u8:f32, 0.2:5>>
}
// CHECK-NOT: quant.scast
// CHECK: "onnx.QuantizeLinear"
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: a_scale = [1.000000e-01
// CHECK-SAME: a_zero_point = [0]
// CHECK-SAME: y_scale = [2.000000e-01
// CHECK-SAME: y_zero_point = [5]

// -----

//===----------------------------------------------------------------------===//
// Pattern 3: Scast node -> Dequantize node (diff scale/zp) -> Requantize + DQ
//===----------------------------------------------------------------------===//

// Test 9: quant.scast(scale1,zp1) -> DequantizeLinear(scale2,zp2) -> Requantize + DQ
// CHECK-LABEL: @scast_then_dq_different_params
func.func @scast_then_dq_different_params(%arg0: tensor<1x4x4x!quant.uniform<u8:f32, 0.1:0>>) -> tensor<1x4x4xf32> {
  %scale_dq = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %zp_dq = onnx.Constant dense<5> : tensor<ui8>
  %0 = quant.scast %arg0 : tensor<1x4x4x!quant.uniform<u8:f32, 0.1:0>> to tensor<1x4x4xui8>
  %1 = "onnx.DequantizeLinear"(%0, %scale_dq, %zp_dq) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4x4xf32>
  return %1 : tensor<1x4x4xf32>
}
// CHECK-NOT: quant.scast
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: a_scale = [1.000000e-01
// CHECK-SAME: a_zero_point = [0]
// CHECK-SAME: y_scale = [2.000000e-01
// CHECK-SAME: y_zero_point = [5]
// CHECK: "onnx.DequantizeLinear"
