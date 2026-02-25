// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// RUN: onnx-mlir-opt --remove-pairs-and-move-down-reshape %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive: remove paired reshapes across a small chain.
// Uses XCompiler custom eltwise op ("onnx.XCOMPILERFusedEltwise") between the
// two reshapes. Both reshapes are identity (types stay consistent after bypass),
// matching the "paired reshape" condition:
//   shape(reshape1.input) == shape(reshape2.output)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_remove_paired_reshape
// CHECK-NOT: "onnx.Reshape"
// CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %{{.*}}) {{.*}}nonlinear = "NONE"{{.*}}type = "ADD"
// CHECK: return %[[FUSED]]
func.func @test_remove_paired_reshape(%arg0: tensor<1x4xui8>) -> tensor<1x4xui8> {
  %shape14 = onnx.Constant dense<[1, 4]> : tensor<2xi64>
  %shape41 = onnx.Constant dense<[4, 1]> : tensor<2xi64>
  %b = onnx.Constant dense<[[1], [1], [1], [1]]> : tensor<4x1xui8>

  %r1 = "onnx.Reshape"(%arg0, %shape41) {allowzero = 0 : si64} : (tensor<1x4xui8>, tensor<2xi64>) -> tensor<4x1xui8>
  %fused = "onnx.XCOMPILERFusedEltwise"(%r1, %b) {type = "ADD", nonlinear = "NONE"} : (tensor<4x1xui8>, tensor<4x1xui8>) -> tensor<4x1xui8>
  %r2 = "onnx.Reshape"(%fused, %shape14) {allowzero = 0 : si64} : (tensor<4x1xui8>, tensor<2xi64>) -> tensor<1x4xui8>
  return %r2 : tensor<1x4xui8>
}

//===----------------------------------------------------------------------===//
// Negative: shapes do not match; keep reshapes.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_no_remove_when_shapes_dont_match
// CHECK: "onnx.Reshape"(%{{.*}}, %{{.*}}) {allowzero = 0 : si64} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
// CHECK: "onnx.Reshape"(%{{.*}}, %{{.*}}) {allowzero = 0 : si64} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<2x2xf32>
// CHECK: "onnx.Reshape"(%{{.*}}, %{{.*}}) {allowzero = 0 : si64} : (tensor<2x2xf32>, tensor<2xi64>) -> tensor<1x4xf32>
func.func @test_no_remove_when_shapes_dont_match(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %shape14 = onnx.Constant dense<[1, 4]> : tensor<2xi64>
  %shape22 = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %c_add = onnx.Constant dense<[[1.0, 1.0, 1.0, 1.0]]> : tensor<1x4xf32>
  %c_zero22 = onnx.Constant dense<[[0.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>

  %r1 = "onnx.Reshape"(%arg0, %shape14) {allowzero = 0 : si64} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  %add = "onnx.Add"(%r1, %c_add) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %r2 = "onnx.Reshape"(%add, %shape22) {allowzero = 0 : si64} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<2x2xf32>
  // Keep the reshapes live (avoid reshape(reshape(x)) folding).
  %add2 = "onnx.Add"(%r2, %c_zero22) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %r3 = "onnx.Reshape"(%add2, %shape14) {allowzero = 0 : si64} : (tensor<2x2xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  return %r3 : tensor<1x4xf32>
}