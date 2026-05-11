// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt %s -onnx-infer-tensornames | FileCheck %s

func.func @reshape_result(%arg0: tensor<1x768x128xf32> {onnx.name = "input"}) -> tensor<1x12x64x128xf32> {
  %0 = onnx.Constant dense<[1, 12, 64, 128]> : tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x768x128xf32>, tensor<4xi64>) -> tensor<1x12x64x128xf32>
  return %1 : tensor<1x12x64x128xf32>
}

// CHECK-LABEL: @reshape_result
// CHECK: onnx.Reshape
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Reshape", [1, 768, 128], [1, 12, 64, 128]]]]

func.func @reshape_operand(%arg0: tensor<1x768x128xf32>) -> tensor<1x12x64x128xf32> {
  %0 = onnx.Constant dense<[1, 12, 64, 128]> : tensor<4xi64>
  %1 = "onnx.Identity"(%arg0) : (tensor<1x768x128xf32>) -> tensor<1x768x128xf32>
  %2 = "onnx.Reshape"(%1, %0) {ResultNames = ["output"], allowzero = 0 : si64} : (tensor<1x768x128xf32>, tensor<4xi64>) -> tensor<1x12x64x128xf32>
  return %2 : tensor<1x12x64x128xf32>
}

// CHECK-LABEL: @reshape_operand
// CHECK: onnx.Identity
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["output", ["Reshape", [1, 12, 64, 128], [1, 768, 128]]]]

func.func @transpose_result(%arg0: tensor<1x128x768xf32> {onnx.name = "input"}) -> tensor<1x768x128xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x128x768xf32>) -> tensor<1x768x128xf32>
  return %0: tensor<1x768x128xf32>
}

// CHECK-LABEL: @transpose_result
// CHECK: onnx.Transpose
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Transpose", [1, 128, 768], [0, 2, 1], [1, 768, 128]]]]

func.func @transpose_operand(%arg0: tensor<1x128x768xf32>) -> tensor<1x768x128xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
  %1 = "onnx.Transpose"(%0) {ResultNames = ["output"], perm = [0, 2, 1]} : (tensor<1x128x768xf32>) -> tensor<1x768x128xf32>
  return %1: tensor<1x768x128xf32>
}

// CHECK-LABEL: @transpose_operand
// CHECK: onnx.Identity
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["output", ["Transpose", [1, 768, 128], [0, 2, 1], [1, 128, 768]]]]

func.func @pad_result(%arg0: tensor<1x3xf32> {onnx.name = "input"}) -> tensor<1x8xf32> {
  %0 = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %1 = onnx.Constant dense<[4, 1]> : tensor<2xi64>
  %2 = onnx.Constant dense<-1> : tensor<1xi64>
  %3 = "onnx.Pad"(%arg0, %1, %0, %2) {mode = "constant"} : (tensor<1x3xf32>, tensor<2xi64>, tensor<f32>, tensor<1xi64>) -> tensor<1x8xf32>
  return %3 : tensor<1x8xf32>
}

// CHECK-LABEL: @pad_result
// CHECK: onnx.Pad
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Pad", [1, 3], [4], [1], [1], dense<0.000000e+00> : tensor<f32>, [1, 8]]]]

func.func @pad_operand(%arg0: tensor<1x3xf32>) -> tensor<1x8xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %1 = onnx.Constant dense<0.0> : tensor<f32>
  %2 = onnx.Constant dense<[4, 1]> : tensor<2xi64>
  %3 = onnx.Constant dense<-1> : tensor<1xi64>
  %4 = "onnx.Pad"(%0, %2, %1, %3) {ResultNames = ["output"]} : (tensor<1x3xf32>, tensor<2xi64>, tensor<f32>, tensor<1xi64>) -> tensor<1x8xf32>
  return %4 : tensor<1x8xf32>
}

// CHECK-LABEL: @pad_operand
// CHECK: onnx.Identity
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["output", ["Slice", [1, 8], [4], [7], [1], [1, 3]]]]
// ResultNames shouldn't be inferred for constants
// CHECK-NEXT: onnx.Constant dense
// CHECK-NEXT: onnx.Constant dense
// CHECK-NEXT: onnx.Constant dense

func.func @slice_result(%arg0: tensor<1x8xf32> {onnx.name = "input"}) -> tensor<1x4xf32> {
  %1 = onnx.Constant dense<4> : tensor<1xi64>
  %2 = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %3 = onnx.Constant dense<-1> : tensor<1xi64>
  %4 = onnx.Constant dense<1> : tensor<1xi64>
  %5 = "onnx.Slice"(%arg0, %1, %2, %3, %4) : (tensor<1x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4xf32>
  return %5 : tensor<1x4xf32>
}

// CHECK-LABEL: @slice_result
// CHECK: onnx.Slice
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Slice", [1, 8], [4], [8], [1], [1, 4]]]]

func.func @slice_operand(%arg0: tensor<1x8xf32>) -> tensor<1x3xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %1 = onnx.Constant dense<2> : tensor<1xi64>
  %2 = onnx.Constant dense<5> : tensor<1xi64>
  %3 = onnx.Constant dense<-1> : tensor<1xi64>
  %4 = onnx.Constant dense<1> : tensor<1xi64>
  %5 = "onnx.Slice"(%0, %1, %2, %3, %4) {ResultNames = ["output"]} : (tensor<1x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3xf32>
  return %5 : tensor<1x3xf32>
}

// CHECK-LABEL: @slice_operand
// CHECK: onnx.Identity
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["output", ["Pad", [1, 3], [2], [3], [1], unit, [1, 8]]]]

func.func @transpose_reshape(%arg0: tensor<1x128x768xf32> {onnx.name = "input"}) -> tensor<1x12x64x128xf32> {
  %0 = onnx.Constant dense<[1, 12, 64, 128]> : tensor<4xi64>
  %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x128x768xf32>) -> tensor<1x768x128xf32>
  %2 = "onnx.Reshape"(%1, %0) {allowzero = 0 : si64} : (tensor<1x768x128xf32>, tensor<4xi64>) -> tensor<1x12x64x128xf32>
  return %2 : tensor<1x12x64x128xf32>
}

// CHECK-LABEL: @transpose_reshape
// CHECK: onnx.Transpose
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Transpose", [1, 128, 768], [0, 2, 1], [1, 768, 128]]]]
// CHECK-NEXT: onnx.Reshape
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Transpose", [1, 128, 768], [0, 2, 1], [1, 768, 128]], ["Reshape", [1, 768, 128], [1, 12, 64, 128]]]]

func.func @conv_double_shielding(%arg0: tensor<1x320x64x64xf32> {onnx.name = "input"}) -> tensor<1x320x64x64xf32> {
  %0 = onnx.Constant {ResultNames = ["bias"]} dense_resource<__elided__> : tensor<320xf32>
  %1 = onnx.Constant {ResultNames = ["wgt"]} dense_resource<__elided__> : tensor<320x320x3x3xf32>
  %2 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x320x64x64xf32>) -> tensor<1x64x64x320xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 1, 2]} : (tensor<1x64x64x320xf32>) -> tensor<1x320x64x64xf32>
  %4 = "onnx.Transpose"(%1) {perm = [3, 2, 1, 0]} : (tensor<320x320x3x3xf32>) -> tensor<3x3x320x320xf32>
  %5 = "onnx.Transpose"(%4) {perm = [3, 2, 1, 0]} : (tensor<3x3x320x320xf32>) -> tensor<320x320x3x3xf32>
  %6 = "onnx.Conv"(%3, %5, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x320x64x64xf32>, tensor<320x320x3x3xf32>, tensor<320xf32>) -> tensor<1x320x64x64xf32>
  %7 = "onnx.Transpose"(%6) {perm = [0, 2, 3, 1]} : (tensor<1x320x64x64xf32>) -> tensor<1x64x64x320xf32>
  %8 = "onnx.Transpose"(%7) {ResultNames = ["conv"], perm = [0, 3, 1, 2]} : (tensor<1x64x64x320xf32>) -> tensor<1x320x64x64xf32>
  return %8 : tensor<1x320x64x64xf32>
}

// CHECK-LABEL: @conv_double_shielding
// CHECK: onnx.Transpose
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Transpose", [1, 320, 64, 64], [0, 2, 3, 1], [1, 64, 64, 320]]]]
// CHECK-NEXT: onnx.Transpose
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Transpose", [1, 320, 64, 64], [0, 2, 3, 1], [1, 64, 64, 320]], ["Transpose", [1, 64, 64, 320], [0, 3, 1, 2], [1, 320, 64, 64]]]]
// CHECK-NEXT: onnx.Transpose
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["wgt", ["Transpose", [320, 320, 3, 3], [3, 2, 1, 0], [3, 3, 320, 320]]]]
// CHECK-NEXT: onnx.Transpose
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["wgt", ["Transpose", [320, 320, 3, 3], [3, 2, 1, 0], [3, 3, 320, 320]], ["Transpose", [3, 3, 320, 320], [3, 2, 1, 0], [320, 320, 3, 3]]]]
// CHECK-NEXT: onnx.Conv
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["conv", ["Transpose", [1, 320, 64, 64], [0, 2, 3, 1], [1, 64, 64, 320]], ["Transpose", [1, 64, 64, 320], [0, 3, 1, 2], [1, 320, 64, 64]]]]
// CHECK-NEXT: onnx.Transpose
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["conv", ["Transpose", [1, 320, 64, 64], [0, 2, 3, 1], [1, 64, 64, 320]]]]
// CHECK-NEXT: onnx.Transpose
// CHECK-SAME: ResultNames = ["conv"]

func.func @tile_result(%arg0: tensor<1x3xf32> {onnx.name = "input"}) -> tensor<2x6xf32> {
  %0 = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %1 = "onnx.Tile"(%arg0, %0) : (tensor<1x3xf32>, tensor<2xi64>) -> tensor<2x6xf32>
  return %1 : tensor<2x6xf32>
}

// CHECK-LABEL: @tile_result
// CHECK: onnx.Tile
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Tile", [1, 3], [2, 2], [2, 6]]]]

func.func @tile_operand(%arg0: tensor<1x4xf32>) -> tensor<3x4xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %1 = onnx.Constant dense<[3, 1]> : tensor<2xi64>
  %2 = "onnx.Tile"(%0, %1) {ResultNames = ["output"]} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %2 : tensor<3x4xf32>
}

// CHECK-LABEL: @tile_operand
// CHECK: onnx.Identity
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["output", ["Slice", [3, 4], [0, 0], [1, 4], [0, 1], [1, 4]]]]
// CHECK-NEXT: onnx.Constant dense
// CHECK-NEXT: onnx.Tile
// CHECK-SAME: ResultNames = ["output"]

func.func @tile_reshape(%arg0: tensor<1x6xf32> {onnx.name = "input"}) -> tensor<4x12xf32> {
  %shape = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
  %1 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} : (tensor<1x6xf32>, tensor<3xi64>) -> tensor<1x2x3xf32>
  %repeats = onnx.Constant dense<[4, 1, 4]> : tensor<3xi64>
  %2 = "onnx.Tile"(%1, %repeats) : (tensor<1x2x3xf32>, tensor<3xi64>) -> tensor<4x2x12xf32>
  %shape2 = onnx.Constant dense<[4, 12, -1]> : tensor<3xi64>
  %3 = "onnx.Reshape"(%2, %shape2) {allowzero = 0 : si64} : (tensor<4x2x12xf32>, tensor<3xi64>) -> tensor<4x12x2xf32>
  %shape3 = onnx.Constant dense<[4, 12]> : tensor<2xi64>
  %4 = "onnx.Reshape"(%2, %shape3) {allowzero = 0 : si64} : (tensor<4x2x12xf32>, tensor<2xi64>) -> tensor<4x12xf32>
  return %4 : tensor<4x12xf32>
}

// CHECK-LABEL: @tile_reshape
// CHECK: onnx.Reshape
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Reshape", [1, 6], [1, 2, 3]]]]
// CHECK:      onnx.Tile
// CHECK-SAME: ResultNames = [
// CHECK-SAME: ["input", ["Reshape", [1, 6], [1, 2, 3]], ["Tile", [1, 2, 3], [4, 1, 4], [4, 2, 12]]]]
