// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --transfer-op-shape-to-4d %s | FileCheck %s

// Define quantization types for testing
!qtype1 = !quant.uniform<i8:f32, 0.5:10>
!qtype2 = !quant.uniform<i8:f32, 0.25:5>
!qtype3 = !quant.uniform<i16:f32, 0.125:0>

// ============================================================================
// Test 1: Element-wise Add with 3D inputs (same shape) -> convert to 4D
// [2,3,4] -> pad with 1 at front -> [1,2,3,4]
// Verifies quant type is preserved through reshape and add
// ============================================================================
func.func @test_add_3d_same_shape(%arg0: tensor<2x3x4x!qtype1>, %arg1: tensor<2x3x4x!qtype1>) -> tensor<2x3x4x!qtype1> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3x4x!qtype1>, tensor<2x3x4x!qtype1>) -> tensor<2x3x4x!qtype1>
  return %0 : tensor<2x3x4x!qtype1>
}
// CHECK-LABEL: func.func @test_add_3d_same_shape
// CHECK-DAG: %[[OUT_SHAPE:.*]] = onnx.Constant dense<[2, 3, 4]> : tensor<3xi64>
// CHECK-DAG: %[[SHAPE4D:.*]] = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xi64>
// CHECK: %[[R1:.*]] = "onnx.Reshape"(%arg0, %[[SHAPE4D]]) {{.*}} : (tensor<2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>, tensor<4xi64>) -> tensor<1x2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>
// CHECK: %[[R2:.*]] = "onnx.Reshape"(%arg1, %[[SHAPE4D]]) {{.*}} : (tensor<2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>, tensor<4xi64>) -> tensor<1x2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>
// CHECK: %[[ADD:.*]] = "onnx.Add"(%[[R1]], %[[R2]]) : (tensor<1x2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>, tensor<1x2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>) -> tensor<1x2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>
// CHECK: %[[OUT:.*]] = "onnx.Reshape"(%[[ADD]], %[[OUT_SHAPE]]) {{.*}} : (tensor<1x2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>, tensor<3xi64>) -> tensor<2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>
// CHECK: return %[[OUT]]

// ============================================================================
// Test 2: Element-wise Mul with broadcasting - 3D x scalar broadcast
// Input1: [2,3,4] (same as output) -> reshaped to [1,2,3,4]
// Input2: [1] (scalar, shape != output) -> passed through as-is 
// Output: [2,3,4] -> [1,2,3,4]
// ============================================================================
func.func @test_mul_broadcast_scalar(%arg0: tensor<2x3x4x!qtype3>, %arg1: tensor<1x!qtype3>) -> tensor<2x3x4x!qtype3> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<2x3x4x!qtype3>, tensor<1x!qtype3>) -> tensor<2x3x4x!qtype3>
  return %0 : tensor<2x3x4x!qtype3>
}
// CHECK-LABEL: func.func @test_mul_broadcast_scalar
// CHECK-DAG: %[[OUT_SHAPE:.*]] = onnx.Constant dense<[2, 3, 4]> : tensor<3xi64>
// CHECK-DAG: %[[SHAPE4D:.*]] = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xi64>
// CHECK: %[[R1:.*]] = "onnx.Reshape"(%arg0, %[[SHAPE4D]]) {{.*}} : (tensor<2x3x4x!quant.uniform<i16:f32, 1.250000e-01>>, tensor<4xi64>) -> tensor<1x2x3x4x!quant.uniform<i16:f32, 1.250000e-01>>
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[R1]], %arg1) : (tensor<1x2x3x4x!quant.uniform<i16:f32, 1.250000e-01>>, tensor<1x!quant.uniform<i16:f32, 1.250000e-01>>) -> tensor<1x2x3x4x!quant.uniform<i16:f32, 1.250000e-01>>
// CHECK: %[[OUT:.*]] = "onnx.Reshape"(%[[MUL]], %[[OUT_SHAPE]]) {{.*}} : (tensor<1x2x3x4x!quant.uniform<i16:f32, 1.250000e-01>>, tensor<3xi64>) -> tensor<2x3x4x!quant.uniform<i16:f32, 1.250000e-01>>
// CHECK: return %[[OUT]]

// ============================================================================
// Test 3: Element-wise Add with 5D inputs (same shape) -> convert to 4D
// [2,3,4,5,6] with batch != 1 -> needs reshaping
// Verifies quant type is preserved through 5D to 4D conversion
// ============================================================================
func.func @test_add_5d_same_shape(%arg0: tensor<2x3x4x5x6x!qtype2>, %arg1: tensor<2x3x4x5x6x!qtype2>) -> tensor<2x3x4x5x6x!qtype2> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3x4x5x6x!qtype2>, tensor<2x3x4x5x6x!qtype2>) -> tensor<2x3x4x5x6x!qtype2>
  return %0 : tensor<2x3x4x5x6x!qtype2>
}
// CHECK-LABEL: func.func @test_add_5d_same_shape
// CHECK: "onnx.Reshape"{{.*}}-> tensor<{{.*}}x!quant.uniform<i8:f32, 2.500000e-01:5>>
// CHECK: "onnx.Reshape"{{.*}}-> tensor<{{.*}}x!quant.uniform<i8:f32, 2.500000e-01:5>>
// CHECK: "onnx.Add"{{.*}}-> tensor<{{.*}}x!quant.uniform<i8:f32, 2.500000e-01:5>>
// CHECK: "onnx.Reshape"{{.*}}-> tensor<2x3x4x5x6x!quant.uniform<i8:f32, 2.500000e-01:5>>
// CHECK: return

// ============================================================================
// Test 4: 4D inputs with batch=1 should NOT be transformed (early return)
// Verifies quant type is preserved when no transformation occurs
// ============================================================================
func.func @test_add_4d_already_valid(%arg0: tensor<1x2x3x4x!qtype1>, %arg1: tensor<1x2x3x4x!qtype1>) -> tensor<1x2x3x4x!qtype1> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x2x3x4x!qtype1>, tensor<1x2x3x4x!qtype1>) -> tensor<1x2x3x4x!qtype1>
  return %0 : tensor<1x2x3x4x!qtype1>
}
// CHECK-LABEL: func.func @test_add_4d_already_valid
// CHECK-NOT: "onnx.Reshape"
// CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1) : (tensor<1x2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>, tensor<1x2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>) -> tensor<1x2x3x4x!quant.uniform<i8:f32, 5.000000e-01:10>>
// CHECK: return %[[ADD]]
