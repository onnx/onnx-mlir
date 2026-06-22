// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file %s --xmc-requantize | FileCheck %s

// =============================================================================
// XmcRequantizePass tests.
// Post-quant-types: ops carry `!quant.uniform<...>` element types directly.
// For Group B data-flow ops with mismatched operand/result quant, the pass
// inserts an XCOMPILERRequantize on the appropriate edge.
// =============================================================================

// -----
// Reshape: operand quant differs from result quant -> retype result to operand
// quant, insert Requantize on the output edge.
func.func @reshape_requantize(%arg0: tensor<1x8x64x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x8x16x4x!quant.uniform<i8:f32, 0.2>> {
  %shape = onnx.Constant dense<[1, 8, 16, 4]> : tensor<4xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} : (tensor<1x8x64x!quant.uniform<i8:f32, 0.1>>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<i8:f32, 0.2>>
  return %0 : tensor<1x8x16x4x!quant.uniform<i8:f32, 0.2>>
}
// CHECK-LABEL: func.func @reshape_requantize
// CHECK: %[[R:.*]] = "onnx.Reshape"(%arg0, %{{.*}})
// CHECK-SAME: -> tensor<1x8x16x4x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK: %[[REQ:.*]] = "onnx.XCOMPILERRequantize"(%[[R]])
// CHECK-SAME: a_scale = [1.000000e-01 : f32]
// CHECK-SAME: y_scale = [2.000000e-01 : f32]
// CHECK-SAME: -> tensor<1x8x16x4x!quant.uniform<i8:f32, 2.000000e-01>>
// CHECK: return %[[REQ]]

// -----
// Transpose: same scale on both sides -> no Requantize inserted.
func.func @transpose_same_scale(%arg0: tensor<1x8x16x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x8x4x16x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 1, 3, 2]} : (tensor<1x8x16x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x8x4x16x!quant.uniform<i8:f32, 0.1>>
  return %0 : tensor<1x8x4x16x!quant.uniform<i8:f32, 0.1>>
}
// CHECK-LABEL: func.func @transpose_same_scale
// CHECK-NOT: onnx.XCOMPILERRequantize
// CHECK: onnx.Transpose

// -----
// Transpose: mismatched scale -> insert Requantize.
func.func @transpose_requantize(%arg0: tensor<1x8x16x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x8x4x16x!quant.uniform<i8:f32, 0.2>> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 1, 3, 2]} : (tensor<1x8x16x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x8x4x16x!quant.uniform<i8:f32, 0.2>>
  return %0 : tensor<1x8x4x16x!quant.uniform<i8:f32, 0.2>>
}
// CHECK-LABEL: func.func @transpose_requantize
// CHECK: %[[T:.*]] = "onnx.Transpose"(%arg0)
// CHECK-SAME: -> tensor<1x8x4x16x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK: %[[REQ:.*]] = "onnx.XCOMPILERRequantize"(%[[T]])
// CHECK-SAME: -> tensor<1x8x4x16x!quant.uniform<i8:f32, 2.000000e-01>>
// CHECK: return %[[REQ]]

// -----
// Slice with mismatched quant -> insert Requantize.
func.func @slice_requantize(%arg0: tensor<10x16x!quant.uniform<i8:f32, 0.05>>) -> tensor<5x16x!quant.uniform<i8:f32, 0.1>> {
  %starts = onnx.Constant dense<[0]> : tensor<1xi64>
  %ends   = onnx.Constant dense<[5]> : tensor<1xi64>
  %axes   = onnx.Constant dense<[0]> : tensor<1xi64>
  %steps  = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<10x16x!quant.uniform<i8:f32, 0.05>>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5x16x!quant.uniform<i8:f32, 0.1>>
  return %0 : tensor<5x16x!quant.uniform<i8:f32, 0.1>>
}
// CHECK-LABEL: func.func @slice_requantize
// CHECK: %[[S:.*]] = "onnx.Slice"(%arg0
// CHECK-SAME: -> tensor<5x16x!quant.uniform<i8:f32, 5.000000e-02>>
// CHECK: %[[REQ:.*]] = "onnx.XCOMPILERRequantize"(%[[S]])
// CHECK-SAME: a_scale = [5.000000e-02 : f32]
// CHECK-SAME: y_scale = [1.000000e-01 : f32]

// -----
// DepthToSpace with mismatched quant -> insert Requantize.
func.func @depth_to_space_requantize(%arg0: tensor<1x12x4x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x3x8x8x!quant.uniform<i8:f32, 0.05>> {
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x12x4x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x3x8x8x!quant.uniform<i8:f32, 0.05>>
  return %0 : tensor<1x3x8x8x!quant.uniform<i8:f32, 0.05>>
}
// CHECK-LABEL: func.func @depth_to_space_requantize
// CHECK: %[[D:.*]] = "onnx.DepthToSpace"(%arg0)
// CHECK-SAME: -> tensor<1x3x8x8x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK: %[[REQ:.*]] = "onnx.XCOMPILERRequantize"(%[[D]])
// CHECK-SAME: -> tensor<1x3x8x8x!quant.uniform<i8:f32, 5.000000e-02>>

// -----
// Squeeze (new op vs OptimizeOnnxRequantizationPass) -> insert Requantize.
func.func @squeeze_requantize(%arg0: tensor<1x8x16x!quant.uniform<i8:f32, 0.1>>) -> tensor<8x16x!quant.uniform<i8:f32, 0.2>> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %0 = "onnx.Squeeze"(%arg0, %axes) : (tensor<1x8x16x!quant.uniform<i8:f32, 0.1>>, tensor<1xi64>) -> tensor<8x16x!quant.uniform<i8:f32, 0.2>>
  return %0 : tensor<8x16x!quant.uniform<i8:f32, 0.2>>
}
// CHECK-LABEL: func.func @squeeze_requantize
// CHECK: %[[S:.*]] = "onnx.Squeeze"(%arg0
// CHECK-SAME: -> tensor<8x16x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK: %[[REQ:.*]] = "onnx.XCOMPILERRequantize"(%[[S]])
// CHECK-SAME: -> tensor<8x16x!quant.uniform<i8:f32, 2.000000e-01>>

// -----
// Unsqueeze (new op) -> insert Requantize.
func.func @unsqueeze_requantize(%arg0: tensor<8x16x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x8x16x!quant.uniform<i8:f32, 0.2>> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %0 = "onnx.Unsqueeze"(%arg0, %axes) : (tensor<8x16x!quant.uniform<i8:f32, 0.1>>, tensor<1xi64>) -> tensor<1x8x16x!quant.uniform<i8:f32, 0.2>>
  return %0 : tensor<1x8x16x!quant.uniform<i8:f32, 0.2>>
}
// CHECK-LABEL: func.func @unsqueeze_requantize
// CHECK: %[[U:.*]] = "onnx.Unsqueeze"(%arg0
// CHECK-SAME: -> tensor<1x8x16x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK: %[[REQ:.*]] = "onnx.XCOMPILERRequantize"(%[[U]])
// CHECK-SAME: -> tensor<1x8x16x!quant.uniform<i8:f32, 2.000000e-01>>

// -----
// Flatten (new op) -> insert Requantize.
func.func @flatten_requantize(%arg0: tensor<2x3x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<6x4x!quant.uniform<i8:f32, 0.2>> {
  %0 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x3x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<6x4x!quant.uniform<i8:f32, 0.2>>
  return %0 : tensor<6x4x!quant.uniform<i8:f32, 0.2>>
}
// CHECK-LABEL: func.func @flatten_requantize
// CHECK: %[[F:.*]] = "onnx.Flatten"(%arg0)
// CHECK-SAME: -> tensor<6x4x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK: %[[REQ:.*]] = "onnx.XCOMPILERRequantize"(%[[F]])
// CHECK-SAME: -> tensor<6x4x!quant.uniform<i8:f32, 2.000000e-01>>

// -----
// Identity (new op) -> insert Requantize.
func.func @identity_requantize(%arg0: tensor<4x8x!quant.uniform<i8:f32, 0.1>>) -> tensor<4x8x!quant.uniform<i8:f32, 0.2>> {
  %0 = "onnx.Identity"(%arg0) : (tensor<4x8x!quant.uniform<i8:f32, 0.1>>) -> tensor<4x8x!quant.uniform<i8:f32, 0.2>>
  return %0 : tensor<4x8x!quant.uniform<i8:f32, 0.2>>
}
// CHECK-LABEL: func.func @identity_requantize
// CHECK: %[[I:.*]] = "onnx.Identity"(%arg0)
// CHECK-SAME: -> tensor<4x8x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK: %[[REQ:.*]] = "onnx.XCOMPILERRequantize"(%[[I]])
// CHECK-SAME: -> tensor<4x8x!quant.uniform<i8:f32, 2.000000e-01>>

// -----
// Concat with two mismatched inputs -> one Requantize per mismatched edge.
func.func @concat_two_mismatched(%arg0: tensor<2x4x!quant.uniform<i8:f32, 0.1>>, %arg1: tensor<2x4x!quant.uniform<i8:f32, 0.2>>) -> tensor<4x4x!quant.uniform<i8:f32, 0.3>> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64} : (tensor<2x4x!quant.uniform<i8:f32, 0.1>>, tensor<2x4x!quant.uniform<i8:f32, 0.2>>) -> tensor<4x4x!quant.uniform<i8:f32, 0.3>>
  return %0 : tensor<4x4x!quant.uniform<i8:f32, 0.3>>
}
// CHECK-LABEL: func.func @concat_two_mismatched
// CHECK: %[[R0:.*]] = "onnx.XCOMPILERRequantize"(%arg0)
// CHECK-SAME: a_scale = [1.000000e-01 : f32]
// CHECK-SAME: y_scale = [3.000000e-01 : f32]
// CHECK: %[[R1:.*]] = "onnx.XCOMPILERRequantize"(%arg1)
// CHECK-SAME: a_scale = [2.000000e-01 : f32]
// CHECK-SAME: y_scale = [3.000000e-01 : f32]
// CHECK: "onnx.Concat"(%[[R0]], %[[R1]])

// -----
// Concat with one matching and one mismatched input -> Requantize only on
// the mismatched input edge.
func.func @concat_one_mismatched(%arg0: tensor<2x4x!quant.uniform<i8:f32, 0.1>>, %arg1: tensor<2x4x!quant.uniform<i8:f32, 0.2>>) -> tensor<4x4x!quant.uniform<i8:f32, 0.2>> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64} : (tensor<2x4x!quant.uniform<i8:f32, 0.1>>, tensor<2x4x!quant.uniform<i8:f32, 0.2>>) -> tensor<4x4x!quant.uniform<i8:f32, 0.2>>
  return %0 : tensor<4x4x!quant.uniform<i8:f32, 0.2>>
}
// CHECK-LABEL: func.func @concat_one_mismatched
// CHECK: %[[R:.*]] = "onnx.XCOMPILERRequantize"(%arg0)
// CHECK-SAME: a_scale = [1.000000e-01 : f32]
// CHECK-SAME: y_scale = [2.000000e-01 : f32]
// CHECK: "onnx.Concat"(%[[R]], %arg1)

// -----
// Concat where all inputs match the result quant -> no Requantize inserted.
func.func @concat_all_match(%arg0: tensor<2x4x!quant.uniform<i8:f32, 0.1>>, %arg1: tensor<2x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<4x4x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64} : (tensor<2x4x!quant.uniform<i8:f32, 0.1>>, tensor<2x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<4x4x!quant.uniform<i8:f32, 0.1>>
  return %0 : tensor<4x4x!quant.uniform<i8:f32, 0.1>>
}
// CHECK-LABEL: func.func @concat_all_match
// CHECK-NOT: onnx.XCOMPILERRequantize
// CHECK: onnx.Concat

// -----
// Per-tensor with same scale but different zero point -> insert Requantize.
func.func @reshape_zp_diff(%arg0: tensor<1x8x64x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x8x16x4x!quant.uniform<i8:f32, 0.1:5>> {
  %shape = onnx.Constant dense<[1, 8, 16, 4]> : tensor<4xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} : (tensor<1x8x64x!quant.uniform<i8:f32, 0.1:0>>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<i8:f32, 0.1:5>>
  return %0 : tensor<1x8x16x4x!quant.uniform<i8:f32, 0.1:5>>
}
// CHECK-LABEL: func.func @reshape_zp_diff
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: a_zero_point = [0]
// CHECK-SAME: y_zero_point = [5]

// -----
// Pad with mismatched quant -> insert Requantize.
func.func @pad_requantize(%arg0: tensor<4x4x!quant.uniform<i8:f32, 0.1>>) -> tensor<6x6x!quant.uniform<i8:f32, 0.2>> {
  %pads = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.Pad"(%arg0, %pads, %0, %1) {mode = "edge"} : (tensor<4x4x!quant.uniform<i8:f32, 0.1>>, tensor<4xi64>, none, none) -> tensor<6x6x!quant.uniform<i8:f32, 0.2>>
  return %2 : tensor<6x6x!quant.uniform<i8:f32, 0.2>>
}
// CHECK-LABEL: func.func @pad_requantize
// CHECK: %[[P:.*]] = "onnx.Pad"(%arg0
// CHECK-SAME: -> tensor<6x6x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK: %[[REQ:.*]] = "onnx.XCOMPILERRequantize"(%[[P]])
// CHECK-SAME: -> tensor<6x6x!quant.uniform<i8:f32, 2.000000e-01>>

// -----
// Per-axis quant operand (weight-shape) -> skip; no Requantize inserted.
func.func @per_axis_skipped(%arg0: tensor<16x3x3x3x!quant.uniform<i8:f32:0, {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6}>>) -> tensor<16x3x3x3x!quant.uniform<i8:f32, 0.5>> {
  %0 = "onnx.Identity"(%arg0) : (tensor<16x3x3x3x!quant.uniform<i8:f32:0, {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6}>>) -> tensor<16x3x3x3x!quant.uniform<i8:f32, 0.5>>
  return %0 : tensor<16x3x3x3x!quant.uniform<i8:f32, 0.5>>
}
// CHECK-LABEL: func.func @per_axis_skipped
// CHECK-NOT: onnx.XCOMPILERRequantize
// CHECK: onnx.Identity
