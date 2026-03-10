// RUN: onnx-mlir-opt %s -fix-neg-scale | FileCheck %s

// i8: negative scale with x=1, zp=0
func.func @fix_neg_scale_i8() -> tensor<f32> {
  %0 = onnx.Constant dense<1> : tensor<i8>
  %1 = onnx.Constant dense<-5.000000e+02> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.DequantizeLinear"(%0, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  return %3 : tensor<f32>
}
// CHECK-LABEL: @fix_neg_scale_i8
// CHECK-DAG: [[X:%.*]] = onnx.Constant dense<0> : tensor<i8>
// CHECK-DAG: [[SCALE:%.*]] = onnx.Constant dense<5.000000e+02> : tensor<f32>
// CHECK-DAG: [[ZP:%.*]] = onnx.Constant dense<1> : tensor<i8>
// CHECK: "onnx.DequantizeLinear"([[X]], [[SCALE]], [[ZP]])

// ui8: negative scale with x=1, zp=0
func.func @fix_neg_scale_ui8() -> tensor<f32> {
  %0 = onnx.Constant dense<1> : tensor<ui8>
  %1 = onnx.Constant dense<-2.500000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<ui8>
  %3 = "onnx.DequantizeLinear"(%0, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui8>, tensor<f32>, tensor<ui8>) -> tensor<f32>
  return %3 : tensor<f32>
}
// CHECK-LABEL: @fix_neg_scale_ui8
// CHECK-DAG: [[X:%.*]] = onnx.Constant dense<0> : tensor<ui8>
// CHECK-DAG: [[SCALE:%.*]] = onnx.Constant dense<2.500000e-01> : tensor<f32>
// CHECK-DAG: [[ZP:%.*]] = onnx.Constant dense<1> : tensor<ui8>
// CHECK: "onnx.DequantizeLinear"([[X]], [[SCALE]], [[ZP]])

// i16: negative scale with x=1, zp=0
func.func @fix_neg_scale_i16() -> tensor<f32> {
  %0 = onnx.Constant dense<1> : tensor<i16>
  %1 = onnx.Constant dense<-1.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i16>
  %3 = "onnx.DequantizeLinear"(%0, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i16>, tensor<f32>, tensor<i16>) -> tensor<f32>
  return %3 : tensor<f32>
}
// CHECK-LABEL: @fix_neg_scale_i16
// CHECK-DAG: [[X:%.*]] = onnx.Constant dense<0> : tensor<i16>
// CHECK-DAG: [[SCALE:%.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG: [[ZP:%.*]] = onnx.Constant dense<1> : tensor<i16>
// CHECK: "onnx.DequantizeLinear"([[X]], [[SCALE]], [[ZP]])

// ui16: negative scale with x=1, zp=0
func.func @fix_neg_scale_ui16() -> tensor<f32> {
  %0 = onnx.Constant dense<1> : tensor<ui16>
  %1 = onnx.Constant dense<-2.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<ui16>
  %3 = "onnx.DequantizeLinear"(%0, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
  return %3 : tensor<f32>
}
// CHECK-LABEL: @fix_neg_scale_ui16
// CHECK-DAG: [[X:%.*]] = onnx.Constant dense<0> : tensor<ui16>
// CHECK-DAG: [[SCALE:%.*]] = onnx.Constant dense<2.000000e-01> : tensor<f32>
// CHECK-DAG: [[ZP:%.*]] = onnx.Constant dense<1> : tensor<ui16>
// CHECK: "onnx.DequantizeLinear"([[X]], [[SCALE]], [[ZP]])

// Zero scale with x=1, zp=0: scale becomes 1.0, x=0, zp=0
func.func @fix_zero_scale_i8() -> tensor<f32> {
  %0 = onnx.Constant dense<1> : tensor<i8>
  %1 = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.DequantizeLinear"(%0, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  return %3 : tensor<f32>
}
// CHECK-LABEL: @fix_zero_scale_i8
// CHECK-DAG: [[X:%.*]] = onnx.Constant dense<0> : tensor<i8>
// CHECK-DAG: [[SCALE:%.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK: "onnx.DequantizeLinear"([[X]], [[SCALE]], [[X]])

// Zero scale with unsigned type
func.func @fix_zero_scale_ui16() -> tensor<f32> {
  %0 = onnx.Constant dense<1> : tensor<ui16>
  %1 = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<ui16>
  %3 = "onnx.DequantizeLinear"(%0, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
  return %3 : tensor<f32>
}
// CHECK-LABEL: @fix_zero_scale_ui16
// CHECK-DAG: [[X:%.*]] = onnx.Constant dense<0> : tensor<ui16>
// CHECK-DAG: [[SCALE:%.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK: "onnx.DequantizeLinear"([[X]], [[SCALE]], [[X]])

// Negative test: positive scale should not be transformed
func.func @no_fix_positive_scale() -> tensor<f32> {
  %0 = onnx.Constant dense<1> : tensor<i8>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.DequantizeLinear"(%0, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  return %3 : tensor<f32>
}
// CHECK-LABEL: @no_fix_positive_scale
// CHECK: onnx.Constant dense<1> : tensor<i8>
// CHECK: onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK: onnx.Constant dense<0> : tensor<i8>

// Negative test: x != 1 should not be transformed
func.func @no_fix_x_not_one() -> tensor<f32> {
  %0 = onnx.Constant dense<5> : tensor<i8>
  %1 = onnx.Constant dense<-5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.DequantizeLinear"(%0, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  return %3 : tensor<f32>
}
// CHECK-LABEL: @no_fix_x_not_one
// CHECK: onnx.Constant dense<5> : tensor<i8>
// CHECK: onnx.Constant dense<-5.000000e-01> : tensor<f32>
// CHECK: onnx.Constant dense<0> : tensor<i8>
