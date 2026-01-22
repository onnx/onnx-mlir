// RUN: flexml-opt %s -standardize-strided-slice | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// CHECK-LABEL: @non_standard_slice

func.func @non_standard_slice(%arg0: tensor<10x20x30x40xf32>) -> (tensor<10x5x30x9xf32>) {
  %starts = onnx.Constant dense<[5, 10]> : tensor<2xi64>
  %ends = onnx.Constant dense<[15, 35]> : tensor<2xi64>
  %axes = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %result = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<10x20x30x40xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<10x5x30x9xf32>
  return %result : tensor<10x5x30x9xf32>
}

// CHECK: onnx.Slice
// CHECK-SAME: tensor<4xi64>
// CHECK-SAME: tensor<4xi64>
// CHECK-SAME: tensor<4xi64>
// CHECK-SAME: tensor<4xi64>

// CHECK-LABEL: @gather_to_slice
func.func @gather_to_slice(%arg0: tensor<10x20x30xf32>) -> (tensor<5x20x30xf32>) {
  %indices = onnx.Constant dense<[1, 2, 3, 4, 5]> : tensor<5xi64>
  %result = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<10x20x30xf32>, tensor<5xi64>) -> tensor<5x20x30xf32>
  return %result : tensor<5x20x30xf32>
}

// CHECK: onnx.Slice
// CHECK-NOT: onnx.Gather
