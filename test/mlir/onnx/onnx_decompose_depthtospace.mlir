// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-depthtospace-decompose %s -split-input-file | FileCheck %s

// -----

// DCR mode: [N, C*bs*bs, H, W] -> Reshape[N,bs,bs,C,H,W] -> Transpose[0,3,4,1,5,2] -> Reshape[N,C,H*bs,W*bs].
func.func @test_depthtospace_dcr(%arg0: tensor<1x12x4x5xf32>) -> tensor<1x3x8x10xf32> {
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x12x4x5xf32>) -> tensor<1x3x8x10xf32>
  onnx.Return %0 : tensor<1x3x8x10xf32>

// CHECK-LABEL:  func.func @test_depthtospace_dcr
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x12x4x5xf32>) -> tensor<1x3x8x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 2, 2, 3, 4, 5]> : tensor<6xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 3, 8, 10]> : tensor<4xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x12x4x5xf32>, tensor<6xi64>) -> tensor<1x2x2x3x4x5xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Transpose"([[VAR_2_]]) {perm = [0, 3, 4, 1, 5, 2]} : (tensor<1x2x2x3x4x5xf32>) -> tensor<1x3x4x2x5x2xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x3x4x2x5x2xf32>, tensor<4xi64>) -> tensor<1x3x8x10xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<1x3x8x10xf32>
}

// -----

// CRD mode: [N, C*bs*bs, H, W] -> Reshape[N,C,bs,bs,H,W] -> Transpose[0,1,4,2,5,3] -> Reshape[N,C,H*bs,W*bs].
func.func @test_depthtospace_crd(%arg0: tensor<1x12x4x5xf32>) -> tensor<1x3x8x10xf32> {
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "CRD"} : (tensor<1x12x4x5xf32>) -> tensor<1x3x8x10xf32>
  onnx.Return %0 : tensor<1x3x8x10xf32>

// CHECK-LABEL:  func.func @test_depthtospace_crd
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x12x4x5xf32>) -> tensor<1x3x8x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 3, 2, 2, 4, 5]> : tensor<6xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 3, 8, 10]> : tensor<4xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x12x4x5xf32>, tensor<6xi64>) -> tensor<1x3x2x2x4x5xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Transpose"([[VAR_2_]]) {perm = [0, 1, 4, 2, 5, 3]} : (tensor<1x3x2x2x4x5xf32>) -> tensor<1x3x4x2x5x2xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x3x4x2x5x2xf32>, tensor<4xi64>) -> tensor<1x3x8x10xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<1x3x8x10xf32>
}

// -----

// Negative: blocksize 1 is a no-op and must not be decomposed; the
// onnx.DepthToSpace op is preserved.
func.func @test_depthtospace_blocksize1(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 1 : si64, mode = "DCR"} : (tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32>
  onnx.Return %0 : tensor<1x3x4x5xf32>

// CHECK-LABEL:  func.func @test_depthtospace_blocksize1
// CHECK-NOT:       onnx.Transpose
// CHECK:           "onnx.DepthToSpace"
}
