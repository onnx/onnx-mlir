// RUN: onnx-mlir-opt --convert-qdq-to-requantize %s --split-input-file | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// Pattern A: FoldEqualQDQ - fold Q(DQ(x)) when scale/zp/storage match
//===----------------------------------------------------------------------===//

// Same scale + same zp + same storage type -> fold.
// CHECK-LABEL: @fold_equal_qdq_same_params
func.func @fold_equal_qdq_same_params(%arg0: tensor<1x4x4xui8>) -> tensor<1x4x4xui8> {
  %s   = onnx.Constant dense<0.5> : tensor<f32>
  %zp  = onnx.Constant dense<10> : tensor<ui8>
  %0   = "onnx.DequantizeLinear"(%arg0, %s, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4x4xf32>
  %1   = "onnx.QuantizeLinear"(%0, %s, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4x4xui8>
  return %1 : tensor<1x4x4xui8>
}
// CHECK-NOT: onnx.QuantizeLinear
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.XCOMPILERRequantize
// CHECK: return %arg0

// -----

//===----------------------------------------------------------------------===//
// Pattern B: InsertRequantizeBetweenQDQ - insert XCOMPILERRequantize on
// the f32 edge between DQ and Q when scale/zp/storage type differ.
//===----------------------------------------------------------------------===//

// Different scale -> insert Requantize on f32 between DQ and Q.
// CHECK-LABEL: @insert_requantize_different_scale
func.func @insert_requantize_different_scale(%arg0: tensor<1x4x4xui8>) -> tensor<1x4x4xui8> {
  %s1  = onnx.Constant dense<0.5> : tensor<f32>
  %zp1 = onnx.Constant dense<10> : tensor<ui8>
  %s2  = onnx.Constant dense<0.25> : tensor<f32>
  %zp2 = onnx.Constant dense<10> : tensor<ui8>
  %0   = "onnx.DequantizeLinear"(%arg0, %s1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4x4xf32>
  %1   = "onnx.QuantizeLinear"(%0, %s2, %zp2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4x4xui8>
  return %1 : tensor<1x4x4xui8>
}
// CHECK: %[[DQ:.+]] = "onnx.DequantizeLinear"
// CHECK: %[[RQ:.+]] = "onnx.XCOMPILERRequantize"(%[[DQ]])
// CHECK-SAME: a_scale = [5.000000e-01 : f32]
// CHECK-SAME: a_zero_point = [10]
// CHECK-SAME: y_scale = [2.500000e-01 : f32]
// CHECK-SAME: y_zero_point = [10]
// CHECK-SAME: (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[Q:.+]] = "onnx.QuantizeLinear"(%[[RQ]],

// -----

// Different zero point -> insert Requantize.
// CHECK-LABEL: @insert_requantize_different_zp
func.func @insert_requantize_different_zp(%arg0: tensor<1x4x4xui8>) -> tensor<1x4x4xui8> {
  %s   = onnx.Constant dense<0.5> : tensor<f32>
  %zp1 = onnx.Constant dense<10> : tensor<ui8>
  %zp2 = onnx.Constant dense<20> : tensor<ui8>
  %0   = "onnx.DequantizeLinear"(%arg0, %s, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4x4xf32>
  %1   = "onnx.QuantizeLinear"(%0, %s, %zp2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4x4xui8>
  return %1 : tensor<1x4x4xui8>
}
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: a_zero_point = [10]
// CHECK-SAME: y_zero_point = [20]

// -----

// Different storage type (ui8 -> ui16) -> insert Requantize.
// CHECK-LABEL: @insert_requantize_different_storage
func.func @insert_requantize_different_storage(%arg0: tensor<1x4x4xui8>) -> tensor<1x4x4xui16> {
  %s1  = onnx.Constant dense<0.5> : tensor<f32>
  %zp1 = onnx.Constant dense<10> : tensor<ui8>
  %s2  = onnx.Constant dense<5.0e-5> : tensor<f32>
  %zp2 = onnx.Constant dense<32000> : tensor<ui16>
  %0   = "onnx.DequantizeLinear"(%arg0, %s1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4x4xf32>
  %1   = "onnx.QuantizeLinear"(%0, %s2, %zp2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4x4xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x4x4xui16>
  return %1 : tensor<1x4x4xui16>
}
// CHECK: "onnx.XCOMPILERRequantize"
// CHECK-SAME: (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>

// -----

//===----------------------------------------------------------------------===//
// Negative: Q without a DQ producer should not be touched.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @no_dq_producer
func.func @no_dq_producer(%arg0: tensor<1x4x4xf32>) -> tensor<1x4x4xui8> {
  %s   = onnx.Constant dense<0.5> : tensor<f32>
  %zp  = onnx.Constant dense<10> : tensor<ui8>
  %0   = "onnx.QuantizeLinear"(%arg0, %s, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4x4xui8>
  return %0 : tensor<1x4x4xui8>
}
// CHECK-NOT: onnx.XCOMPILERRequantize
// CHECK: onnx.QuantizeLinear
