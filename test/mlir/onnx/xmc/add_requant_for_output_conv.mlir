// RUN: onnx-mlir-opt --add-requant-for-output-conv %s --split-input-file | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// Positive tests: scast should be REPLACED by a no-op XCOMPILERRequantize
// whose output is the same storage type the scast was producing, so the
// DequantizeLinear consumes the requantize directly.
//===----------------------------------------------------------------------===//

// Test 1: XFEConv producer with multi-fanout (scast/DQ + another consumer).
// The conv's quant result is consumed by both the output-edge scast and a
// second XCOMPILERRequantize for the compute branch; this pattern is the
// canonical shape from real models (see F2_voe_10611_x3_p3/onnx.mlir).
// CHECK-LABEL: @xfeconv_multi_fanout_output
func.func @xfeconv_multi_fanout_output(
    %arg0: tensor<1x1x480x2x!quant.uniform<u8:f32, 0.05:128>>,
    %weight: tensor<32x1x7x2x!quant.uniform<i8:f32, 0.02>>,
    %bias: tensor<32x!quant.uniform<i32:f32, 1.000000e-03>>,
    %dq_scale: tensor<f32>, %dq_zp: tensor<ui8>)
    -> (tensor<1x1x240x32xf32>,
        tensor<1x1x240x32x!quant.uniform<u8:f32, 0.15:116>>) {
  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [1, 7], pads = [0, 0, 0, 5],
       strides = [1, 2]}
      : (tensor<1x1x480x2x!quant.uniform<u8:f32, 0.05:128>>,
         tensor<32x1x7x2x!quant.uniform<i8:f32, 0.02>>,
         tensor<32x!quant.uniform<i32:f32, 1.000000e-03>>)
      -> tensor<1x1x240x32x!quant.uniform<u8:f32, 0.125:128>>

  // Output-edge: scast -> DQ -> f32
  %scast = quant.scast %conv
      : tensor<1x1x240x32x!quant.uniform<u8:f32, 0.125:128>>
      to tensor<1x1x240x32xui8>
  %dq = "onnx.DequantizeLinear"(%scast, %dq_scale, %dq_zp)
      {axis = 1 : si64, block_size = 0 : si64}
      : (tensor<1x1x240x32xui8>, tensor<f32>, tensor<ui8>)
      -> tensor<1x1x240x32xf32>

  // Compute branch: existing XCOMPILERRequantize with different y params.
  %rq_compute = "onnx.XCOMPILERRequantize"(%conv)
      {a_scale = [0.125 : f32], a_zero_point = [128],
       y_scale = [0.15 : f32], y_zero_point = [116]}
      : (tensor<1x1x240x32x!quant.uniform<u8:f32, 0.125:128>>)
      -> tensor<1x1x240x32x!quant.uniform<u8:f32, 0.15:116>>

  return %dq, %rq_compute
      : tensor<1x1x240x32xf32>,
        tensor<1x1x240x32x!quant.uniform<u8:f32, 0.15:116>>
}
// The output-edge scast is replaced by a placeholder XCOMPILERRequantize
// that produces the same storage type (ui8) and carries dummy a/y attrs
// (scale=1.0, zp=0) -- matches xcompiler's flow; downstream passes will
// fill the real values in.
// CHECK: %[[CONV:.*]] = "onnx.XFEConv"
// CHECK-NOT: quant.scast %[[CONV]]
// CHECK: %[[RQ_OUT:.*]] = "onnx.XCOMPILERRequantize"(%[[CONV]])
// CHECK-SAME: a_scale = [1.000000e+00 : f32]
// CHECK-SAME: a_zero_point = [0]
// CHECK-SAME: y_scale = [1.000000e+00 : f32]
// CHECK-SAME: y_zero_point = [0]
// CHECK-SAME: -> tensor<1x1x240x32xui8>
// CHECK: "onnx.DequantizeLinear"(%[[RQ_OUT]]
// The existing compute-branch XCOMPILERRequantize still consumes %conv
// and keeps its original (real) a/y attrs.
// CHECK: "onnx.XCOMPILERRequantize"(%[[CONV]])
// CHECK-SAME: y_scale = [1.500000e-01 : f32]
// CHECK-SAME: y_zero_point = [116]

// -----

// Test 2: ResultNames propagation. Producer's ResultNames must end up on
// the new XCOMPILERRequantize (mirrors what
// ResultNamesUpdater::notifyOperationReplaced does).
// CHECK-LABEL: @xfeconv_propagates_result_names
func.func @xfeconv_propagates_result_names(
    %arg0: tensor<1x1x480x2x!quant.uniform<u8:f32, 0.05:128>>,
    %weight: tensor<32x1x7x2x!quant.uniform<i8:f32, 0.02>>,
    %bias: tensor<32x!quant.uniform<i32:f32, 1.000000e-03>>,
    %dq_scale: tensor<f32>, %dq_zp: tensor<ui8>)
    -> (tensor<1x1x240x32xf32>,
        tensor<1x1x240x32x!quant.uniform<u8:f32, 0.15:116>>) {
  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {ResultNames = ["out_menc00_QuantizeLinear_Output"],
       activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [1, 7], pads = [0, 0, 0, 5],
       strides = [1, 2]}
      : (tensor<1x1x480x2x!quant.uniform<u8:f32, 0.05:128>>,
         tensor<32x1x7x2x!quant.uniform<i8:f32, 0.02>>,
         tensor<32x!quant.uniform<i32:f32, 1.000000e-03>>)
      -> tensor<1x1x240x32x!quant.uniform<u8:f32, 0.125:128>>

  %scast = quant.scast %conv
      : tensor<1x1x240x32x!quant.uniform<u8:f32, 0.125:128>>
      to tensor<1x1x240x32xui8>
  %dq = "onnx.DequantizeLinear"(%scast, %dq_scale, %dq_zp)
      {axis = 1 : si64, block_size = 0 : si64}
      : (tensor<1x1x240x32xui8>, tensor<f32>, tensor<ui8>)
      -> tensor<1x1x240x32xf32>

  %rq_compute = "onnx.XCOMPILERRequantize"(%conv)
      {a_scale = [0.125 : f32], a_zero_point = [128],
       y_scale = [0.15 : f32], y_zero_point = [116]}
      : (tensor<1x1x240x32x!quant.uniform<u8:f32, 0.125:128>>)
      -> tensor<1x1x240x32x!quant.uniform<u8:f32, 0.15:116>>

  return %dq, %rq_compute
      : tensor<1x1x240x32xf32>,
        tensor<1x1x240x32x!quant.uniform<u8:f32, 0.15:116>>
}
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: ResultNames = ["out_menc00_QuantizeLinear_Output"]

// -----

// Test 3: XCOMPILERFusedEltwise producer with multi-fanout.
// CHECK-LABEL: @fused_eltwise_multi_fanout_output
func.func @fused_eltwise_multi_fanout_output(
    %a: tensor<1x4x4x16x!quant.uniform<u8:f32, 0.04:128>>,
    %none: none,
    %dq_scale: tensor<f32>, %dq_zp: tensor<ui8>)
    -> (tensor<1x4x4x16xf32>,
        tensor<1x4x4x16x!quant.uniform<u8:f32, 0.05:128>>) {
  %e = "onnx.XCOMPILERFusedEltwise"(%a, %none)
      {enable_lut_sigmoid = false, leakyrelu_alpha = 2.000000e-01 : f32,
       nonlinear = "NONE", prelu_in = 51 : si64, prelu_shift = 8 : si64,
       type = "LEAKYRELU"}
      : (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.04:128>>, none)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.03:64>>

  %scast = quant.scast %e
      : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.03:64>>
      to tensor<1x4x4x16xui8>
  %dq = "onnx.DequantizeLinear"(%scast, %dq_scale, %dq_zp)
      {axis = 1 : si64, block_size = 0 : si64}
      : (tensor<1x4x4x16xui8>, tensor<f32>, tensor<ui8>)
      -> tensor<1x4x4x16xf32>

  %rq_compute = "onnx.XCOMPILERRequantize"(%e)
      {a_scale = [0.03 : f32], a_zero_point = [64],
       y_scale = [0.05 : f32], y_zero_point = [128]}
      : (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.03:64>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.05:128>>

  return %dq, %rq_compute
      : tensor<1x4x4x16xf32>,
        tensor<1x4x4x16x!quant.uniform<u8:f32, 0.05:128>>
}
// CHECK: %[[E:.*]] = "onnx.XCOMPILERFusedEltwise"
// CHECK-NOT: quant.scast %[[E]]
// CHECK: %[[RQ_OUT:.*]] = "onnx.XCOMPILERRequantize"(%[[E]])
// CHECK-SAME: a_scale = [1.000000e+00 : f32]
// CHECK-SAME: a_zero_point = [0]
// CHECK-SAME: y_scale = [1.000000e+00 : f32]
// CHECK-SAME: y_zero_point = [0]
// CHECK-SAME: -> tensor<1x4x4x16xui8>
// CHECK: "onnx.DequantizeLinear"(%[[RQ_OUT]]

// -----

// Test 4: XCOMPILERDepthwiseConv producer with multi-fanout.
// CHECK-LABEL: @depthwise_conv_multi_fanout_output
func.func @depthwise_conv_multi_fanout_output(
    %arg0: tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>>,
    %weight: tensor<16x3x3x1x!quant.uniform<i8:f32, 0.01>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 4.000000e-04>>,
    %dq_scale: tensor<f32>, %dq_zp: tensor<ui8>)
    -> (tensor<1x8x8x16xf32>,
        tensor<1x8x8x16x!quant.uniform<u8:f32, 0.06:128>>) {
  %conv = "onnx.XCOMPILERDepthwiseConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 16 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1],
       strides = [1, 1]}
      : (tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>>,
         tensor<16x3x3x1x!quant.uniform<i8:f32, 0.01>>,
         tensor<16x!quant.uniform<i32:f32, 4.000000e-04>>)
      -> tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:140>>

  %scast = quant.scast %conv
      : tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:140>>
      to tensor<1x8x8x16xui8>
  %dq = "onnx.DequantizeLinear"(%scast, %dq_scale, %dq_zp)
      {axis = 1 : si64, block_size = 0 : si64}
      : (tensor<1x8x8x16xui8>, tensor<f32>, tensor<ui8>)
      -> tensor<1x8x8x16xf32>

  %rq_compute = "onnx.XCOMPILERRequantize"(%conv)
      {a_scale = [0.05 : f32], a_zero_point = [140],
       y_scale = [0.06 : f32], y_zero_point = [128]}
      : (tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:140>>)
      -> tensor<1x8x8x16x!quant.uniform<u8:f32, 0.06:128>>

  return %dq, %rq_compute
      : tensor<1x8x8x16xf32>,
        tensor<1x8x8x16x!quant.uniform<u8:f32, 0.06:128>>
}
// CHECK: %[[DW:.*]] = "onnx.XCOMPILERDepthwiseConv"
// CHECK-NOT: quant.scast %[[DW]]
// CHECK: %[[RQ_OUT:.*]] = "onnx.XCOMPILERRequantize"(%[[DW]])
// CHECK-SAME: a_scale = [1.000000e+00 : f32]
// CHECK-SAME: a_zero_point = [0]
// CHECK-SAME: y_scale = [1.000000e+00 : f32]
// CHECK-SAME: y_zero_point = [0]
// CHECK-SAME: -> tensor<1x8x8x16xui8>
// CHECK: "onnx.DequantizeLinear"(%[[RQ_OUT]]

// -----

//===----------------------------------------------------------------------===//
// Negative tests: should NOT touch the IR.
//===----------------------------------------------------------------------===//

// Test 5: Single use (only the scast/DQ consumes the producer's quant
// result) -- no multi-fanout, so no rewrite.
// CHECK-LABEL: @xfeconv_single_use_no_insert
func.func @xfeconv_single_use_no_insert(
    %arg0: tensor<1x1x480x2x!quant.uniform<u8:f32, 0.05:128>>,
    %weight: tensor<32x1x7x2x!quant.uniform<i8:f32, 0.02>>,
    %bias: tensor<32x!quant.uniform<i32:f32, 1.000000e-03>>,
    %dq_scale: tensor<f32>, %dq_zp: tensor<ui8>)
    -> tensor<1x1x240x32xf32> {
  %conv = "onnx.XFEConv"(%arg0, %weight, %bias)
      {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1],
       group = 1 : si64, kernel_shape = [1, 7], pads = [0, 0, 0, 5],
       strides = [1, 2]}
      : (tensor<1x1x480x2x!quant.uniform<u8:f32, 0.05:128>>,
         tensor<32x1x7x2x!quant.uniform<i8:f32, 0.02>>,
         tensor<32x!quant.uniform<i32:f32, 1.000000e-03>>)
      -> tensor<1x1x240x32x!quant.uniform<u8:f32, 0.125:128>>

  %scast = quant.scast %conv
      : tensor<1x1x240x32x!quant.uniform<u8:f32, 0.125:128>>
      to tensor<1x1x240x32xui8>
  %dq = "onnx.DequantizeLinear"(%scast, %dq_scale, %dq_zp)
      {axis = 1 : si64, block_size = 0 : si64}
      : (tensor<1x1x240x32xui8>, tensor<f32>, tensor<ui8>)
      -> tensor<1x1x240x32xf32>

  return %dq : tensor<1x1x240x32xf32>
}
// scast is preserved; no XCOMPILERRequantize is added.
// CHECK: "onnx.XFEConv"
// CHECK-NOT: "onnx.XCOMPILERRequantize"
// CHECK: quant.scast
// CHECK: "onnx.DequantizeLinear"

// -----

// Test 6: No scast layer between producer and DQ -- producer feeds DQ
// directly with a storage-typed value. Should NOT match (post-QuantTypes
// canonical shape always has a scast).
// CHECK-LABEL: @no_scast_layer_no_insert
func.func @no_scast_layer_no_insert(
    %x: tensor<1x4x4x16xui8>,
    %dq_scale: tensor<f32>, %dq_zp: tensor<ui8>)
    -> tensor<1x4x4x16xf32> {
  %dq = "onnx.DequantizeLinear"(%x, %dq_scale, %dq_zp)
      {axis = 1 : si64, block_size = 0 : si64}
      : (tensor<1x4x4x16xui8>, tensor<f32>, tensor<ui8>)
      -> tensor<1x4x4x16xf32>
  return %dq : tensor<1x4x4x16xf32>
}
// CHECK-NOT: "onnx.XCOMPILERRequantize"
// CHECK: "onnx.DequantizeLinear"

// -----

// Test 7: Non-allow-listed producer of the scast (XCOMPILERRequantize feeds
// the scast -- excluded by design to keep the rewriter idempotent). This is
// also the post-rewrite shape (a same-shape input from a previous pass),
// so it doubles as the idempotency check: a second pass over the IR must
// produce no further changes.
// CHECK-LABEL: @existing_requantize_producer_no_insert
func.func @existing_requantize_producer_no_insert(
    %x: tensor<1x4x4x16x!quant.uniform<u8:f32, 0.04:128>>,
    %dq_scale: tensor<f32>, %dq_zp: tensor<ui8>)
    -> (tensor<1x4x4x16xf32>,
        tensor<1x4x4x16x!quant.uniform<u8:f32, 0.05:128>>) {
  %rq = "onnx.XCOMPILERRequantize"(%x)
      {a_scale = [0.04 : f32], a_zero_point = [128],
       y_scale = [0.04 : f32], y_zero_point = [128]}
      : (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.04:128>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.04:128>>

  %scast = quant.scast %rq
      : tensor<1x4x4x16x!quant.uniform<u8:f32, 0.04:128>>
      to tensor<1x4x4x16xui8>
  %dq = "onnx.DequantizeLinear"(%scast, %dq_scale, %dq_zp)
      {axis = 1 : si64, block_size = 0 : si64}
      : (tensor<1x4x4x16xui8>, tensor<f32>, tensor<ui8>)
      -> tensor<1x4x4x16xf32>

  %rq_compute = "onnx.XCOMPILERRequantize"(%x)
      {a_scale = [0.04 : f32], a_zero_point = [128],
       y_scale = [0.05 : f32], y_zero_point = [128]}
      : (tensor<1x4x4x16x!quant.uniform<u8:f32, 0.04:128>>)
      -> tensor<1x4x4x16x!quant.uniform<u8:f32, 0.05:128>>

  return %dq, %rq_compute
      : tensor<1x4x4x16xf32>,
        tensor<1x4x4x16x!quant.uniform<u8:f32, 0.05:128>>
}
// Only the two pre-existing XCOMPILERRequantize ops should remain (the
// pass must not insert a third) and the scast still exists, untouched.
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK: quant.scast
// CHECK: "onnx.DequantizeLinear"
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-NOT: "onnx.XCOMPILERRequantize"
