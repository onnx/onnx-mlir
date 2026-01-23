// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --eliminate-reshape-around-slice %s | FileCheck %s

module {
  // ============================================================================
  // Test 1: Pattern 0 - Single Eltwise (right branch only)
  //
  //              reshape_fix_0
  //             /             \
  //        slice_0           slice_1
  //           |                 |
  //      reshape_fix_1    eltwise_0 ← const_fix
  //                             |
  //                       reshape_fix_2
  // ============================================================================

  func.func @eliminate_reshape_single_eltwise(%arg0: tensor<1x8x1x1xf32>) -> (tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>) {
    // reshape_fix_0: [1, 8, 1, 1] -> [1, 1, 8, 1]
    %shape0 = "onnx.Constant"() {value = dense<[1, 1, 8, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %reshape0 = "onnx.Reshape"(%arg0, %shape0) {allowzero = 0 : si64} : (tensor<1x8x1x1xf32>, tensor<4xi64>) -> tensor<1x1x8x1xf32>

    // slice_0: takes first half [0:4] on dim 2
    %starts0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
    %ends0 = "onnx.Constant"() {value = dense<[1, 1, 4, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %axes0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
    %steps0 = "onnx.Constant"() {value = dense<[1, 1, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %slice0 = "onnx.Slice"(%reshape0, %starts0, %ends0, %axes0, %steps0) : (tensor<1x1x8x1xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x1x4x1xf32>

    // reshape_fix_1: [1, 1, 4, 1] -> [1, 4, 1, 1]
    %shape1 = "onnx.Constant"() {value = dense<[1, 4, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %reshape1 = "onnx.Reshape"(%slice0, %shape1) {allowzero = 0 : si64} : (tensor<1x1x4x1xf32>, tensor<4xi64>) -> tensor<1x4x1x1xf32>

    // slice_1: takes second half [4:8] on dim 2
    %starts1 = "onnx.Constant"() {value = dense<[0, 0, 4, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
    %ends1 = "onnx.Constant"() {value = dense<[1, 1, 8, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %slice1 = "onnx.Slice"(%reshape0, %starts1, %ends1, %axes0, %steps0) : (tensor<1x1x8x1xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x1x4x1xf32>

    // const_fix for eltwise
    %const_bias = "onnx.Constant"() {value = dense<1.0> : tensor<1x1x4x1xf32>} : () -> tensor<1x1x4x1xf32>

    // eltwise_0: Add slice_1 + const
    %eltwise = "onnx.Add"(%slice1, %const_bias) : (tensor<1x1x4x1xf32>, tensor<1x1x4x1xf32>) -> tensor<1x1x4x1xf32>

    // reshape_fix_2: [1, 1, 4, 1] -> [1, 4, 1, 1]
    %shape2 = "onnx.Constant"() {value = dense<[1, 4, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %reshape2 = "onnx.Reshape"(%eltwise, %shape2) {allowzero = 0 : si64} : (tensor<1x1x4x1xf32>, tensor<4xi64>) -> tensor<1x4x1x1xf32>

    return %reshape1, %reshape2 : tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>
  }
  // CHECK-LABEL: func.func @eliminate_reshape_single_eltwise
  // CHECK: "onnx.Slice"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (tensor<1x8x1x1xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x1x1xf32>
  // CHECK: "onnx.Slice"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (tensor<1x8x1x1xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x1x1xf32>
  // CHECK: "onnx.Add"({{.*}}, {{.*}}) : (tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x1x1xf32>

  // ============================================================================
  // Test 2: Pattern 1 - Double Eltwise in Series (right branch)
  // ============================================================================

  func.func @eliminate_reshape_double_eltwise(%arg0: tensor<1x8x1x1xf32>) -> (tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>) {
    // reshape_fix_0: [1, 8, 1, 1] -> [1, 1, 8, 1]
    %shape0 = "onnx.Constant"() {value = dense<[1, 1, 8, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %reshape0 = "onnx.Reshape"(%arg0, %shape0) {allowzero = 0 : si64} : (tensor<1x8x1x1xf32>, tensor<4xi64>) -> tensor<1x1x8x1xf32>

    // slice_0: takes first half [0:4] on dim 2
    %starts0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
    %ends0 = "onnx.Constant"() {value = dense<[1, 1, 4, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %axes0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
    %steps0 = "onnx.Constant"() {value = dense<[1, 1, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %slice0 = "onnx.Slice"(%reshape0, %starts0, %ends0, %axes0, %steps0) : (tensor<1x1x8x1xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x1x4x1xf32>

    // reshape_fix_1: [1, 1, 4, 1] -> [1, 4, 1, 1]
    %shape1 = "onnx.Constant"() {value = dense<[1, 4, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %reshape1 = "onnx.Reshape"(%slice0, %shape1) {allowzero = 0 : si64} : (tensor<1x1x4x1xf32>, tensor<4xi64>) -> tensor<1x4x1x1xf32>

    // slice_1: takes second half [4:8] on dim 2
    %starts1 = "onnx.Constant"() {value = dense<[0, 0, 4, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
    %ends1 = "onnx.Constant"() {value = dense<[1, 1, 8, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %slice1 = "onnx.Slice"(%reshape0, %starts1, %ends1, %axes0, %steps0) : (tensor<1x1x8x1xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x1x4x1xf32>

    // eltwise_0: Mul with scale (single tensor input pattern)
    %const_scale = "onnx.Constant"() {value = dense<2.0> : tensor<1x1x4x1xf32>} : () -> tensor<1x1x4x1xf32>
    %eltwise0 = "onnx.Mul"(%slice1, %const_scale) : (tensor<1x1x4x1xf32>, tensor<1x1x4x1xf32>) -> tensor<1x1x4x1xf32>

    // eltwise_1: Add with bias
    %const_bias = "onnx.Constant"() {value = dense<1.0> : tensor<1x1x4x1xf32>} : () -> tensor<1x1x4x1xf32>
    %eltwise1 = "onnx.Add"(%eltwise0, %const_bias) : (tensor<1x1x4x1xf32>, tensor<1x1x4x1xf32>) -> tensor<1x1x4x1xf32>

    // reshape_fix_2: [1, 1, 4, 1] -> [1, 4, 1, 1]
    %shape2 = "onnx.Constant"() {value = dense<[1, 4, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %reshape2 = "onnx.Reshape"(%eltwise1, %shape2) {allowzero = 0 : si64} : (tensor<1x1x4x1xf32>, tensor<4xi64>) -> tensor<1x4x1x1xf32>

    return %reshape1, %reshape2 : tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>
  }
  // CHECK-LABEL: func.func @eliminate_reshape_double_eltwise
  // CHECK: "onnx.Slice"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (tensor<1x8x1x1xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x1x1xf32>
  // CHECK: "onnx.Slice"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (tensor<1x8x1x1xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x1x1xf32>
  // CHECK: "onnx.Mul"({{.*}}, {{.*}}) : (tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x1x1xf32>
  // CHECK: "onnx.Add"({{.*}}, {{.*}}) : (tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x1x1xf32>

  // ============================================================================
  // Test 3: Pattern with Quantized Types
  // ============================================================================

  func.func @eliminate_reshape_double_eltwise_quant(%arg0: tensor<1x8x1x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>) -> (tensor<1x4x1x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>, tensor<1x4x1x1x!quant.uniform<u8:f32, 0.30000001192092896:64>>) {
    // reshape_fix_0: [1, 8, 1, 1] -> [1, 1, 8, 1]
    %shape0 = "onnx.Constant"() {value = dense<[1, 1, 8, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %reshape0 = "onnx.Reshape"(%arg0, %shape0) {allowzero = 0 : si64} : (tensor<1x8x1x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>, tensor<4xi64>) -> tensor<1x1x8x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>

    // slice_0: takes first half [0:4] on dim 2
    %starts0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
    %ends0 = "onnx.Constant"() {value = dense<[1, 1, 4, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %axes0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
    %steps0 = "onnx.Constant"() {value = dense<[1, 1, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %slice0 = "onnx.Slice"(%reshape0, %starts0, %ends0, %axes0, %steps0) : (tensor<1x1x8x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x1x4x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>

    // reshape_fix_1: [1, 1, 4, 1] -> [1, 4, 1, 1]
    %shape1 = "onnx.Constant"() {value = dense<[1, 4, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %reshape1 = "onnx.Reshape"(%slice0, %shape1) {allowzero = 0 : si64} : (tensor<1x1x4x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>, tensor<4xi64>) -> tensor<1x4x1x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>

    // slice_1: takes second half [4:8] on dim 2
    %starts1 = "onnx.Constant"() {value = dense<[0, 0, 4, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
    %ends1 = "onnx.Constant"() {value = dense<[1, 1, 8, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %slice1 = "onnx.Slice"(%reshape0, %starts1, %ends1, %axes0, %steps0) : (tensor<1x1x8x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x1x4x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>

    // eltwise_0: Mul with quant scale (different quant params)
    %const_scale = "onnx.Constant"() {value = dense<128> : tensor<1x1x4x1xui8>} : () -> tensor<1x1x4x1x!quant.uniform<u8:f32, 0.20000000298023224:0>>
    %eltwise0 = "onnx.Mul"(%slice1, %const_scale) : (tensor<1x1x4x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>, tensor<1x1x4x1x!quant.uniform<u8:f32, 0.20000000298023224:0>>) -> tensor<1x1x4x1x!quant.uniform<u8:f32, 0.30000001192092896:64>>

    // eltwise_1: Add with quant bias (different quant params)
    %const_bias = "onnx.Constant"() {value = dense<32> : tensor<1x1x4x1xui8>} : () -> tensor<1x1x4x1x!quant.uniform<u8:f32, 0.15000000596046448:32>>
    %eltwise1 = "onnx.Add"(%eltwise0, %const_bias) : (tensor<1x1x4x1x!quant.uniform<u8:f32, 0.30000001192092896:64>>, tensor<1x1x4x1x!quant.uniform<u8:f32, 0.15000000596046448:32>>) -> tensor<1x1x4x1x!quant.uniform<u8:f32, 0.30000001192092896:64>>

    // reshape_fix_2: [1, 1, 4, 1] -> [1, 4, 1, 1]
    %shape2 = "onnx.Constant"() {value = dense<[1, 4, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %reshape2 = "onnx.Reshape"(%eltwise1, %shape2) {allowzero = 0 : si64} : (tensor<1x1x4x1x!quant.uniform<u8:f32, 0.30000001192092896:64>>, tensor<4xi64>) -> tensor<1x4x1x1x!quant.uniform<u8:f32, 0.30000001192092896:64>>

    return %reshape1, %reshape2 : tensor<1x4x1x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>, tensor<1x4x1x1x!quant.uniform<u8:f32, 0.30000001192092896:64>>
  }
  // CHECK-LABEL: func.func @eliminate_reshape_double_eltwise_quant
  // CHECK: "onnx.Slice"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (tensor<1x8x1x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x1x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>
  // CHECK: "onnx.Slice"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (tensor<1x8x1x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x1x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>
  // CHECK: "onnx.Mul"({{.*}}, {{.*}}) : (tensor<1x4x1x1x!quant.uniform<u8:f32, 0.10000000149011612:128>>, tensor<1x4x1x1x!quant.uniform<u8:f32, 0.20000000298023224>>) -> tensor<1x4x1x1x!quant.uniform<u8:f32, 0.30000001192092896:64>>
  // CHECK: "onnx.Add"({{.*}}, {{.*}}) : (tensor<1x4x1x1x!quant.uniform<u8:f32, 0.30000001192092896:64>>, tensor<1x4x1x1x!quant.uniform<u8:f32, 0.15000000596046448:32>>) -> tensor<1x4x1x1x!quant.uniform<u8:f32, 0.30000001192092896:64>>
}

