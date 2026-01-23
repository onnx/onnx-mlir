// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --transform-5d-transpose-to-4d %s | FileCheck %s

module {
  // Test 1: perm = [0, 2, 1, 3, 4] - consecutive pair at positions 3,4 (values 3,4)
  // Merge dims 3,4 -> 4D perm [0, 2, 1, 3]
  func.func @test_perm_02134(%arg0: tensor<1x64x8x7x56xf32>) -> tensor<1x8x64x7x56xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3, 4]} : (tensor<1x64x8x7x56xf32>) -> tensor<1x8x64x7x56xf32>
    return %0 : tensor<1x8x64x7x56xf32>
  }
  // CHECK-LABEL: func.func @test_perm_02134
  // CHECK: onnx.Reshape
  // CHECK: "onnx.Transpose"{{.*}}{perm = [0, 2, 1, 3]}{{.*}}tensor<1x64x8x392xf32>{{.*}}tensor<1x8x64x392xf32>
  // CHECK: onnx.Reshape

  // Test 2: perm = [0, 4, 3, 1, 2] - consecutive pair at positions 3,4 (values 1,2)
  // Merge dims 1,2 -> 4D perm [0, 3, 2, 1]
  func.func @test_perm_04312(%arg0: tensor<1x4x6x32x64xf32>) -> tensor<1x64x32x4x6xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 4, 3, 1, 2]} : (tensor<1x4x6x32x64xf32>) -> tensor<1x64x32x4x6xf32>
    return %0 : tensor<1x64x32x4x6xf32>
  }
  // CHECK-LABEL: func.func @test_perm_04312
  // CHECK: onnx.Reshape
  // CHECK: "onnx.Transpose"{{.*}}{perm = [0, 3, 2, 1]}{{.*}}tensor<1x24x32x64xf32>{{.*}}tensor<1x64x32x24xf32>
  // CHECK: onnx.Reshape

  // Test 3: perm = [0, 1, 2, 4, 3] - consecutive pair at positions 0,1 (values 0,1)
  // Merge dims 0,1 -> 4D perm [0, 1, 3, 2]
  func.func @test_perm_01243(%arg0: tensor<1x3x4x16x32xf32>) -> tensor<1x3x4x32x16xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 1, 2, 4, 3]} : (tensor<1x3x4x16x32xf32>) -> tensor<1x3x4x32x16xf32>
    return %0 : tensor<1x3x4x32x16xf32>
  }
  // CHECK-LABEL: func.func @test_perm_01243
  // CHECK: onnx.Reshape
  // CHECK: "onnx.Transpose"{{.*}}{perm = [0, 1, 3, 2]}{{.*}}tensor<3x4x16x32xf32>{{.*}}tensor<3x4x32x16xf32>
  // CHECK: onnx.Reshape

  // Test 4: perm = [0, 1, 4, 3, 2] - consecutive pair at positions 0,1 (values 0,1)
  // Merge dims 0,1 -> 4D perm [0, 3, 2, 1]
  func.func @test_perm_01432(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x6x5x4xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 1, 4, 3, 2]} : (tensor<2x3x4x5x6xf32>) -> tensor<2x3x6x5x4xf32>
    return %0 : tensor<2x3x6x5x4xf32>
  }
  // CHECK-LABEL: func.func @test_perm_01432
  // CHECK: onnx.Reshape
  // CHECK: "onnx.Transpose"{{.*}}{perm = [0, 3, 2, 1]}{{.*}}tensor<6x4x5x6xf32>{{.*}}tensor<6x6x5x4xf32>
  // CHECK: onnx.Reshape

  // Test 5: perm = [4, 3, 2, 1, 0] - NO consecutive pair, should NOT transform
  func.func @test_no_consecutive(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<6x5x4x3x2xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [4, 3, 2, 1, 0]} : (tensor<2x3x4x5x6xf32>) -> tensor<6x5x4x3x2xf32>
    return %0 : tensor<6x5x4x3x2xf32>
  }
  // CHECK-LABEL: func.func @test_no_consecutive
  // CHECK: "onnx.Transpose"(%arg0) {perm = [4, 3, 2, 1, 0]}
  // CHECK-NOT: onnx.Reshape

  // Test 6: perm = [0, 2, 1, 3, 4] with quantized type
  // Merge dims 3,4 -> 4D perm [0, 2, 1, 3]
  func.func @test_perm_02134_quant(%arg0: tensor<1x64x8x7x56x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<1x8x64x7x56x!quant.uniform<u8:f32, 0.20000000298023224:1>> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3, 4]} : (tensor<1x64x8x7x56x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<1x8x64x7x56x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    return %0 : tensor<1x8x64x7x56x!quant.uniform<u8:f32, 0.20000000298023224:1>>
  }
  // CHECK-LABEL: func.func @test_perm_02134_quant
  // CHECK: onnx.Reshape
  // CHECK: "onnx.Transpose"{{.*}}{perm = [0, 2, 1, 3]}
  // CHECK: onnx.Reshape

  // Test 7: perm = [0, 4, 3, 1, 2] with quantized type
  // Merge dims 1,2 -> 4D perm [0, 3, 2, 1]
  func.func @test_perm_04312_quant(%arg0: tensor<1x4x6x32x64x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<1x64x32x4x6x!quant.uniform<u8:f32, 0.20000000298023224:1>> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 4, 3, 1, 2]} : (tensor<1x4x6x32x64x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<1x64x32x4x6x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    return %0 : tensor<1x64x32x4x6x!quant.uniform<u8:f32, 0.20000000298023224:1>>
  }
  // CHECK-LABEL: func.func @test_perm_04312_quant
  // CHECK: onnx.Reshape
  // CHECK: "onnx.Transpose"{{.*}}{perm = [0, 3, 2, 1]}
  // CHECK: onnx.Reshape
}

