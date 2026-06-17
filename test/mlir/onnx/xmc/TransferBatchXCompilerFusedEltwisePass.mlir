// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --transfer-batch-xcompiler-fused-eltwise %s | FileCheck %s

// -----
// Batch > 1 on fused eltwise: reshape NCHW-style operands to [1,N,C,H*W], eltwise, reshape back.
func.func @batch_fused_add(%arg0: tensor<16x16x300x4x!quant.uniform<i8:f32, 0.01:0>>, %arg1: tensor<16x1x300x4x!quant.uniform<i8:f32, 0.01:0>>) -> tensor<16x16x300x4x!quant.uniform<i8:f32, 0.01:0>> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {type = "ADD", nonlinear = "NONE", enable_lut_sigmoid = false} : (tensor<16x16x300x4x!quant.uniform<i8:f32, 0.01:0>>, tensor<16x1x300x4x!quant.uniform<i8:f32, 0.01:0>>) -> tensor<16x16x300x4x!quant.uniform<i8:f32, 0.01:0>>
  return %0 : tensor<16x16x300x4x!quant.uniform<i8:f32, 0.01:0>>
}

// CHECK-LABEL: func.func @batch_fused_add
// CHECK-DAG: %[[SHAPE1:.*]] = onnx.Constant dense<[1, 16, 16, 1200]> : tensor<4xi64>
// CHECK-DAG: %[[SHAPE2:.*]] = onnx.Constant dense<[1, 16, 1, 1200]> : tensor<4xi64>
// CHECK-DAG: %[[OUTSHAPE:.*]] = onnx.Constant dense<[16, 16, 300, 4]> : tensor<4xi64>
// CHECK: %[[R0:.*]] = "onnx.Reshape"(%arg0, %[[SHAPE1]])
// CHECK: %[[R1:.*]] = "onnx.Reshape"(%arg1, %[[SHAPE2]])
// CHECK: %[[E:.*]] = "onnx.XCOMPILERFusedEltwise"(%[[R0]], %[[R1]])
// CHECK-DAG: type = "ADD"
// CHECK-DAG: nonlinear = "NONE"
// CHECK: %[[OUT:.*]] = "onnx.Reshape"(%[[E]], %[[OUTSHAPE]])
// CHECK: return %[[OUT]]

// -----
// Leading batch is 1: no rewrite.
func.func @no_batch(%arg0: tensor<1x16x300x4x!quant.uniform<i8:f32, 0.01:0>>, %arg1: tensor<1x1x300x4x!quant.uniform<i8:f32, 0.01:0>>) -> tensor<1x16x300x4x!quant.uniform<i8:f32, 0.01:0>> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {type = "ADD", nonlinear = "NONE", enable_lut_sigmoid = false} : (tensor<1x16x300x4x!quant.uniform<i8:f32, 0.01:0>>, tensor<1x1x300x4x!quant.uniform<i8:f32, 0.01:0>>) -> tensor<1x16x300x4x!quant.uniform<i8:f32, 0.01:0>>
  return %0 : tensor<1x16x300x4x!quant.uniform<i8:f32, 0.01:0>>
}

// CHECK-LABEL: func.func @no_batch
// CHECK-NOT: "onnx.Reshape"
// CHECK: "onnx.XCOMPILERFusedEltwise"
// CHECK: return
