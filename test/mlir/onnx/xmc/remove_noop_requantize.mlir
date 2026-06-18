// RUN: onnx-mlir-opt --split-input-file --remove-noop-requantize %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// No-op requantize (input quant == output quant): removed, and its ResultNames
// carry over to the producer.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @noop_requantize_removed
func.func @noop_requantize_removed(
    %arg0: tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>,
    %w: tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>,
    %b: none
) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>> {
  %0 = "onnx.XFEConv"(%arg0, %w, %b) {ResultNames = ["conv_out"], activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>, tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>, none) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>
  %1 = "onnx.XCOMPILERRequantize"(%0) {ResultNames = ["requant_out"], a_scale = [5.000000e-01 : f32], a_zero_point = [0], y_scale = [5.000000e-01 : f32], y_zero_point = [0]} : (tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>
  return %1 : tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>

  // CHECK: "onnx.XFEConv"
  // CHECK-SAME: ResultNames = ["requant_out"]
  // CHECK-NOT: onnx.XCOMPILERRequantize
}

// -----

//===----------------------------------------------------------------------===//
// Attrs say no-op (a_scale==y_scale, a_zp==y_zp) but the quant TYPES differ
// (input 0.25 vs output 0.5). The pass keys off getScale()/getZeroPoint() of
// the types, not the attributes, so this is a real requantize and is kept.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @attrs_noop_but_types_differ_kept
func.func @attrs_noop_but_types_differ_kept(
    %arg0: tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>,
    %w: tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>,
    %b: none
) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>> {
  %0 = "onnx.XFEConv"(%arg0, %w, %b) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>, tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>, none) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.25:0>>
  // Attributes claim a no-op (0.5 == 0.5), but the input type is 0.25 and the
  // output type is 0.5, so the requantize is NOT a no-op.
  %1 = "onnx.XCOMPILERRequantize"(%0) {a_scale = [5.000000e-01 : f32], a_zero_point = [0], y_scale = [5.000000e-01 : f32], y_zero_point = [0]} : (tensor<1x8x8x4x!quant.uniform<i8:f32, 0.25:0>>) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>
  return %1 : tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>

  // CHECK: onnx.XCOMPILERRequantize
}

// -----

//===----------------------------------------------------------------------===//
// Scale change (0.25 -> 0.5): real requantize, kept.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @scale_change_kept
func.func @scale_change_kept(
    %arg0: tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>,
    %w: tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>,
    %b: none
) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>> {
  %0 = "onnx.XFEConv"(%arg0, %w, %b) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>, tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>, none) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.25:0>>
  %1 = "onnx.XCOMPILERRequantize"(%0) {a_scale = [2.500000e-01 : f32], a_zero_point = [0], y_scale = [5.000000e-01 : f32], y_zero_point = [0]} : (tensor<1x8x8x4x!quant.uniform<i8:f32, 0.25:0>>) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>
  return %1 : tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>

  // CHECK: onnx.XCOMPILERRequantize
}

// -----

//===----------------------------------------------------------------------===//
// Zero-point mismatch (0 -> 5): kept.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @zp_mismatch_kept
func.func @zp_mismatch_kept(
    %arg0: tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>,
    %w: tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>,
    %b: none
) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:5>> {
  %0 = "onnx.XFEConv"(%arg0, %w, %b) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>, tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>, none) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>
  %1 = "onnx.XCOMPILERRequantize"(%0) {a_scale = [5.000000e-01 : f32], a_zero_point = [0], y_scale = [5.000000e-01 : f32], y_zero_point = [5]} : (tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:5>>
  return %1 : tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:5>>

  // CHECK: onnx.XCOMPILERRequantize
}

// -----

//===----------------------------------------------------------------------===//
// Storage dtype change (i8 -> i16, same scale/zp): real requantize, kept.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @dtype_change_kept
func.func @dtype_change_kept(
    %arg0: tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>,
    %w: tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>,
    %b: none
) -> tensor<1x8x8x4x!quant.uniform<i16:f32, 0.5:0>> {
  %0 = "onnx.XFEConv"(%arg0, %w, %b) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>, tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>, none) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>
  %1 = "onnx.XCOMPILERRequantize"(%0) {a_scale = [5.000000e-01 : f32], a_zero_point = [0], y_scale = [5.000000e-01 : f32], y_zero_point = [0]} : (tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x8x8x4x!quant.uniform<i16:f32, 0.5:0>>
  return %1 : tensor<1x8x8x4x!quant.uniform<i16:f32, 0.5:0>>

  // CHECK: onnx.XCOMPILERRequantize
}

// -----

//===----------------------------------------------------------------------===//
// No-op requantize feeding a conv: removed, and the consuming conv reads the
// producer conv directly (requantize ResultNames carried over to the producer).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @noop_requantize_into_conv
func.func @noop_requantize_into_conv(
    %arg0: tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>,
    %w0: tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>,
    %w1: tensor<8x1x1x4x!quant.uniform<i8:f32, 0.0078125>>,
    %b: none
) -> tensor<1x8x8x8x!quant.uniform<i8:f32, 0.5:0>> {
  %0 = "onnx.XFEConv"(%arg0, %w0, %b) {ResultNames = ["conv0_out"], activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>, tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>, none) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>
  %1 = "onnx.XCOMPILERRequantize"(%0) {ResultNames = ["requant_out"], a_scale = [5.000000e-01 : f32], a_zero_point = [0], y_scale = [5.000000e-01 : f32], y_zero_point = [0]} : (tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>
  %2 = "onnx.XFEConv"(%1, %w1, %b) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<8x1x1x4x!quant.uniform<i8:f32, 0.0078125>>, none) -> tensor<1x8x8x8x!quant.uniform<i8:f32, 0.5:0>>
  return %2 : tensor<1x8x8x8x!quant.uniform<i8:f32, 0.5:0>>

  // CHECK: %[[C0:.*]] = "onnx.XFEConv"(%arg0,
  // CHECK-SAME: ResultNames = ["requant_out"]
  // CHECK: "onnx.XFEConv"(%[[C0]],
  // CHECK-NOT: onnx.XCOMPILERRequantize
}

// -----

//===----------------------------------------------------------------------===//
// Multi-use producer: requantize input has more than one consumer, kept.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @multi_use_producer_kept
func.func @multi_use_producer_kept(
    %arg0: tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>,
    %w: tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>,
    %b: none
) -> (tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>) {
  %0 = "onnx.XFEConv"(%arg0, %w, %b) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x8x8x16x!quant.uniform<i8:f32, 0.25:0>>, tensor<4x1x1x16x!quant.uniform<i8:f32, 0.0078125>>, none) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>
  %1 = "onnx.XCOMPILERRequantize"(%0) {a_scale = [5.000000e-01 : f32], a_zero_point = [0], y_scale = [5.000000e-01 : f32], y_zero_point = [0]} : (tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>
  return %1, %0 : tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<1x8x8x4x!quant.uniform<i8:f32, 0.5:0>>

  // CHECK: onnx.XCOMPILERRequantize
}
