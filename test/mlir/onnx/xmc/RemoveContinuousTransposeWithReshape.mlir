// RUN: onnx-mlir-opt --split-input-file --remove-continuous-transpose-with-reshape %s | FileCheck %s

// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

// ============================================================================
// SUCCESS CASE 1: Transpose-Reshape-Transpose replaced by single Reshape
// Input shape != Output shape, so a reshape is inserted
// ============================================================================
// CHECK-LABEL: @replace_with_reshape
func.func @replace_with_reshape(%arg0: tensor<1x3x4x5xf32>) -> (tensor<1x3x20xf32>) {
    %0 = onnx.Constant dense<0.0392156877> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<[1, 20, 3]> : tensor<3xi64>
    %3 = "onnx.QuantizeLinear"(%arg0, %0, %1) : (tensor<1x3x4x5xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3x4x5x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %4 = "onnx.Transpose"(%3) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x5x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x4x5x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %5 = "onnx.Reshape"(%4, %2) : (tensor<1x4x5x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<3xi64>) -> tensor<1x20x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %6 = "onnx.Transpose"(%5) {perm = [0, 2, 1]} : (tensor<1x20x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x3x20x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %7 = "onnx.DequantizeLinear"(%6, %0, %1) : (tensor<1x3x20x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<f32>, tensor<ui8>) -> tensor<1x3x20xf32>
    return %7 : tensor<1x3x20xf32>
  }
// CHECK-NOT: onnx.Transpose
// CHECK-DAG: [[SHAPE1:%[a-zA-Z0-9_]+]] = onnx.Constant dense<[1, 3, 20]> : tensor<3xi64>
// CHECK-DAG: [[QUANT1:%[a-zA-Z0-9_]+]] = "onnx.QuantizeLinear"
// CHECK: "onnx.Reshape"([[QUANT1]], [[SHAPE1]])
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.Reshape
// CHECK: return {{.*}} : tensor<1x3x20xf32>

// ============================================================================
// SUCCESS CASE 2: Transpose-Reshape-Transpose removed completely
// Input shape == Output shape, so no ops needed (direct replacement)
// ============================================================================
// CHECK-LABEL: @remove_completely
func.func @remove_completely(%arg0: tensor<60x1xf32>) -> (tensor<60x1xf32>) {
    %0 = onnx.Constant dense<0.0392156877> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<[1, 60]> : tensor<2xi64>
    %3 = "onnx.QuantizeLinear"(%arg0, %0, %1) : (tensor<60x1xf32>, tensor<f32>, tensor<ui8>) -> tensor<60x1x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    // Transpose: 60x1 -> 1x60
    %4 = "onnx.Transpose"(%3) {perm = [1, 0]} : (tensor<60x1x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x60x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    // Reshape: 1x60 -> 1x60 (same shape)
    %5 = "onnx.Reshape"(%4, %2) : (tensor<1x60x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<2xi64>) -> tensor<1x60x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    // Transpose: 1x60 -> 60x1
    %6 = "onnx.Transpose"(%5) {perm = [1, 0]} : (tensor<1x60x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<60x1x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %7 = "onnx.DequantizeLinear"(%6, %0, %1) : (tensor<60x1x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<f32>, tensor<ui8>) -> tensor<60x1xf32>
    return %7 : tensor<60x1xf32>
  }
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.Reshape
// CHECK: [[QUANT2:%[a-zA-Z0-9_]+]] = "onnx.QuantizeLinear"
// CHECK-NEXT: "onnx.DequantizeLinear"([[QUANT2]]
// CHECK: return {{.*}} : tensor<60x1xf32>

// ============================================================================
// SUCCESS CASE 3: Another replace with reshape (different dimensions)
// 4D -> 3D transformation
// ============================================================================
// CHECK-LABEL: @replace_with_reshape_4d_to_3d
func.func @replace_with_reshape_4d_to_3d(%arg0: tensor<2x4x3x5xf32>) -> (tensor<2x4x15xf32>) {
    %0 = onnx.Constant dense<0.0392156877> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<[2, 15, 4]> : tensor<3xi64>
    %3 = "onnx.QuantizeLinear"(%arg0, %0, %1) : (tensor<2x4x3x5xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x4x3x5x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    // perm=[0,2,3,1]: 2x4x3x5 -> 2x3x5x4, consecutive dims 2,3 in perm
    %4 = "onnx.Transpose"(%3) {perm = [0, 2, 3, 1]} : (tensor<2x4x3x5x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<2x3x5x4x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    // Reshape: 2x3x5x4 -> 2x15x4 (merge 3x5 into 15)
    %5 = "onnx.Reshape"(%4, %2) : (tensor<2x3x5x4x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<3xi64>) -> tensor<2x15x4x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    // perm=[0,2,1]: 2x15x4 -> 2x4x15
    %6 = "onnx.Transpose"(%5) {perm = [0, 2, 1]} : (tensor<2x15x4x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<2x4x15x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %7 = "onnx.DequantizeLinear"(%6, %0, %1) : (tensor<2x4x15x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<f32>, tensor<ui8>) -> tensor<2x4x15xf32>
    return %7 : tensor<2x4x15xf32>
  }
// CHECK-NOT: onnx.Transpose
// CHECK-DAG: [[SHAPE3:%[a-zA-Z0-9_]+]] = onnx.Constant dense<[2, 4, 15]> : tensor<3xi64>
// CHECK-DAG: [[QUANT3:%[a-zA-Z0-9_]+]] = "onnx.QuantizeLinear"
// CHECK: "onnx.Reshape"([[QUANT3]], [[SHAPE3]])
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.Reshape
// CHECK: return {{.*}} : tensor<2x4x15xf32>

// ============================================================================
// SUCCESS CASE 3: Non-quantized types
// ============================================================================
// CHECK-LABEL: @pass_non_quantized
func.func @pass_non_quantized(%arg0: tensor<1x3x4x5xf32>) -> (tensor<1x3x20xf32>) {
    %2 = onnx.Constant dense<[1, 20, 3]> : tensor<3xi64>
    // No quantization - using f32 directly
    %4 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x5xf32>) -> tensor<1x4x5x3xf32>
    %5 = "onnx.Reshape"(%4, %2) : (tensor<1x4x5x3xf32>, tensor<3xi64>) -> tensor<1x20x3xf32>
    %6 = "onnx.Transpose"(%5) {perm = [0, 2, 1]} : (tensor<1x20x3xf32>) -> tensor<1x3x20xf32>
    return %6 : tensor<1x3x20xf32>
  }

// CHECK-NOT: onnx.Transpose
// CHECK: [[SHAPE:%[a-zA-Z0-9_]+]] = onnx.Constant dense<[1, 3, 20]> : tensor<3xi64>
// CHECK: "onnx.Reshape"(%arg0, [[SHAPE3]])
// CHECK-NOT: onnx.Transpose
// CHECK: return {{.*}} : tensor<1x3x20xf32>

// ============================================================================
// FAILURE CASE 2: Multiple uses of intermediate result - pattern should NOT match
// ============================================================================
// CHECK-LABEL: @fail_multiple_uses
func.func @fail_multiple_uses(%arg0: tensor<1x3x4x5xf32>) -> (tensor<1x3x20xf32>, tensor<1x20x3xf32>) {
    %0 = onnx.Constant dense<0.0392156877> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<[1, 20, 3]> : tensor<3xi64>
    %3 = "onnx.QuantizeLinear"(%arg0, %0, %1) : (tensor<1x3x4x5xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3x4x5x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %4 = "onnx.Transpose"(%3) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x5x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x4x5x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %5 = "onnx.Reshape"(%4, %2) : (tensor<1x4x5x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<3xi64>) -> tensor<1x20x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %6 = "onnx.Transpose"(%5) {perm = [0, 2, 1]} : (tensor<1x20x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x3x20x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %7 = "onnx.DequantizeLinear"(%6, %0, %1) : (tensor<1x3x20x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<f32>, tensor<ui8>) -> tensor<1x3x20xf32>
    // Second use of reshape result - prevents optimization
    %8 = "onnx.DequantizeLinear"(%5, %0, %1) : (tensor<1x20x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<f32>, tensor<ui8>) -> tensor<1x20x3xf32>
    return %7, %8 : tensor<1x3x20xf32>, tensor<1x20x3xf32>
  }
// Pattern should NOT fire - transposes and reshape should remain
// CHECK: "onnx.Transpose"
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Transpose"

// ============================================================================
// FAILURE CASE 3: Non-inverse permutations - pattern should NOT match
// Transpose1 perm is NOT the inverse of Transpose0 perm (after merging)
// ============================================================================
// CHECK-LABEL: @fail_non_inverse_perm
func.func @fail_non_inverse_perm(%arg0: tensor<1x3x4x5xf32>) -> (tensor<1x20x3xf32>) {
    %0 = onnx.Constant dense<0.0392156877> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<[1, 20, 3]> : tensor<3xi64>
    %3 = "onnx.QuantizeLinear"(%arg0, %0, %1) : (tensor<1x3x4x5xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3x4x5x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %4 = "onnx.Transpose"(%3) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x5x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x4x5x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %5 = "onnx.Reshape"(%4, %2) : (tensor<1x4x5x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<3xi64>) -> tensor<1x20x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    // perm=[0,1,2] is identity, NOT inverse of [0,2,1]
    %6 = "onnx.Transpose"(%5) {perm = [0, 1, 2]} : (tensor<1x20x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x20x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %7 = "onnx.DequantizeLinear"(%6, %0, %1) : (tensor<1x20x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<f32>, tensor<ui8>) -> tensor<1x20x3xf32>
    return %7 : tensor<1x20x3xf32>
  }
// Pattern should NOT fire - transposes and reshape should remain
// CHECK: "onnx.Transpose"
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Transpose"
// CHECK: return {{.*}} : tensor<1x20x3xf32>

// ============================================================================
// FAILURE CASE 4: Merged shapes not equal - pattern should NOT match
// The shape transformation doesn't preserve merged dimensions
// ============================================================================
// CHECK-LABEL: @fail_merged_shapes_not_equal
func.func @fail_merged_shapes_not_equal(%arg0: tensor<1x3x4x5xf32>) -> (tensor<1x6x10xf32>) {
    %0 = onnx.Constant dense<0.0392156877> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<[1, 10, 6]> : tensor<3xi64>
    %3 = "onnx.QuantizeLinear"(%arg0, %0, %1) : (tensor<1x3x4x5xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3x4x5x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %4 = "onnx.Transpose"(%3) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x5x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x4x5x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    // Reshape to different merged shape: 1x10x6 instead of 1x20x3
    %5 = "onnx.Reshape"(%4, %2) : (tensor<1x4x5x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<3xi64>) -> tensor<1x10x6x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %6 = "onnx.Transpose"(%5) {perm = [0, 2, 1]} : (tensor<1x10x6x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x6x10x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %7 = "onnx.DequantizeLinear"(%6, %0, %1) : (tensor<1x6x10x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<f32>, tensor<ui8>) -> tensor<1x6x10xf32>
    return %7 : tensor<1x6x10xf32>
  }
// Pattern should NOT fire - transposes and reshape should remain
// CHECK: "onnx.Transpose"
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Transpose"
// CHECK: return {{.*}} : tensor<1x6x10xf32>

// ============================================================================
// FAILURE CASE 5: Missing reshape between transposes - pattern should NOT match
// ============================================================================
// CHECK-LABEL: @fail_no_reshape
func.func @fail_no_reshape(%arg0: tensor<1x3x4xf32>) -> (tensor<1x3x4xf32>) {
    %0 = onnx.Constant dense<0.0392156877> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %3 = "onnx.QuantizeLinear"(%arg0, %0, %1) : (tensor<1x3x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3x4x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %4 = "onnx.Transpose"(%3) {perm = [0, 2, 1]} : (tensor<1x3x4x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x4x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    // Direct transpose without reshape in between
    %6 = "onnx.Transpose"(%4) {perm = [0, 2, 1]} : (tensor<1x4x3x!quant.uniform<u8:f32, 0.023529412224888802:64>>) -> tensor<1x3x4x!quant.uniform<u8:f32, 0.023529412224888802:64>>
    %7 = "onnx.DequantizeLinear"(%6, %0, %1) : (tensor<1x3x4x!quant.uniform<u8:f32, 0.023529412224888802:64>>, tensor<f32>, tensor<ui8>) -> tensor<1x3x4xf32>
    return %7 : tensor<1x3x4xf32>
  }
// Pattern should NOT fire - this is a different pattern (consecutive transposes)
// CHECK: "onnx.Transpose"
// CHECK: "onnx.Transpose"
// CHECK: return {{.*}} : tensor<1x3x4xf32>
