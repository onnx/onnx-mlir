// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --enable-keepdims-canonicalization=true --canonicalize="test-convergence=true" %s -split-input-file | FileCheck %s

// -----

func.func @reducemin_full_tensor(%arg0: tensor<1x1x1024x1024xbf16>) -> tensor<1x1x1x1xbf16> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceMin"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x1024x1024xbf16>, none) -> tensor<1x1x1x1xbf16>
  onnx.Return %0 : tensor<1x1x1x1xbf16>
}
// CHECK-LABEL:  func.func @reducemin_full_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1024x1024xbf16>) -> tensor<1x1x1x1xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceMin"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x1024x1024xbf16>, tensor<2xi64>) -> tensor<1x1x1x1xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x1x1x1xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemin_negative_axes(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
  %axes = onnx.Constant dense<[-3, -2]> : tensor<2xi64>
  %0 = "onnx.ReduceMin"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<2xi64>) -> tensor<1x1x4xbf16>
  onnx.Return %0 : tensor<1x1x4xbf16>
}
// CHECK-LABEL:  func.func @reducemin_negative_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
// CHECK-NEXT:           onnx.Return [[PARAM_0_]] : tensor<1x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemin_empty_default(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x1xbf16> {
  %axes = onnx.Constant dense<[]> : tensor<0xi64>
  %0 = "onnx.ReduceMin"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<0xi64>) -> tensor<1x1x1xbf16>
  onnx.Return %0 : tensor<1x1x1xbf16>
}
// CHECK-LABEL:  func.func @reducemin_empty_default
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x1xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceMin"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<1xi64>) -> tensor<1x1x1xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x1x1xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemin_empty_noop(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
  %axes = onnx.Constant dense<[]> : tensor<0xi64>
  %0 = "onnx.ReduceMin"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x1x4xbf16>, tensor<0xi64>) -> tensor<1x1x4xbf16>
  onnx.Return %0 : tensor<1x1x4xbf16>
}
// CHECK-LABEL:  func.func @reducemin_empty_noop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
// CHECK-NEXT:           onnx.Return [[PARAM_0_]] : tensor<1x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemin_nonunit(%arg0: tensor<2x3x4xbf16>) -> tensor<2x1x4xbf16> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.ReduceMin"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xbf16>, tensor<1xi64>) -> tensor<2x1x4xbf16>
  onnx.Return %0 : tensor<2x1x4xbf16>
}
// CHECK-LABEL:  func.func @reducemin_nonunit
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xbf16>) -> tensor<2x1x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceMin"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xbf16>, tensor<1xi64>) -> tensor<2x1x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<2x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemin_dynamic(%arg0: tensor<1x?x4xbf16>) -> tensor<1x?x4xbf16> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %0 = "onnx.ReduceMin"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x?x4xbf16>, tensor<1xi64>) -> tensor<1x?x4xbf16>
  onnx.Return %0 : tensor<1x?x4xbf16>
}
// CHECK-LABEL:  func.func @reducemin_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x4xbf16>) -> tensor<1x?x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceMin"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x?x4xbf16>, tensor<1xi64>) -> tensor<1x?x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x?x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemin_keepdims_zero(%arg0: tensor<1x1x4xbf16>) -> tensor<1x4xbf16> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %0 = "onnx.ReduceMin"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<1xi64>) -> tensor<1x4xbf16>
  onnx.Return %0 : tensor<1x4xbf16>
}
// CHECK-LABEL:  func.func @reducemin_keepdims_zero
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 4]> : tensor<2xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1x4xbf16>, tensor<2xi64>) -> tensor<1x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemax_full_tensor(%arg0: tensor<1x1x1024x1024xbf16>) -> tensor<1x1x1x1xbf16> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x1024x1024xbf16>, none) -> tensor<1x1x1x1xbf16>
  onnx.Return %0 : tensor<1x1x1x1xbf16>
}
// CHECK-LABEL:  func.func @reducemax_full_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1024x1024xbf16>) -> tensor<1x1x1x1xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceMax"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x1024x1024xbf16>, tensor<2xi64>) -> tensor<1x1x1x1xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x1x1x1xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemax_negative_axes(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
  %axes = onnx.Constant dense<[-3, -2]> : tensor<2xi64>
  %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<2xi64>) -> tensor<1x1x4xbf16>
  onnx.Return %0 : tensor<1x1x4xbf16>
}
// CHECK-LABEL:  func.func @reducemax_negative_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
// CHECK-NEXT:           onnx.Return [[PARAM_0_]] : tensor<1x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemax_empty_default(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x1xbf16> {
  %axes = onnx.Constant dense<[]> : tensor<0xi64>
  %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<0xi64>) -> tensor<1x1x1xbf16>
  onnx.Return %0 : tensor<1x1x1xbf16>
}
// CHECK-LABEL:  func.func @reducemax_empty_default
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x1xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceMax"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<1xi64>) -> tensor<1x1x1xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x1x1xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemax_empty_noop(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
  %axes = onnx.Constant dense<[]> : tensor<0xi64>
  %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x1x4xbf16>, tensor<0xi64>) -> tensor<1x1x4xbf16>
  onnx.Return %0 : tensor<1x1x4xbf16>
}
// CHECK-LABEL:  func.func @reducemax_empty_noop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
// CHECK-NEXT:           onnx.Return [[PARAM_0_]] : tensor<1x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemax_nonunit(%arg0: tensor<2x3x4xbf16>) -> tensor<2x1x4xbf16> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xbf16>, tensor<1xi64>) -> tensor<2x1x4xbf16>
  onnx.Return %0 : tensor<2x1x4xbf16>
}
// CHECK-LABEL:  func.func @reducemax_nonunit
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xbf16>) -> tensor<2x1x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceMax"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xbf16>, tensor<1xi64>) -> tensor<2x1x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<2x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemax_dynamic(%arg0: tensor<1x?x4xbf16>) -> tensor<1x?x4xbf16> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x?x4xbf16>, tensor<1xi64>) -> tensor<1x?x4xbf16>
  onnx.Return %0 : tensor<1x?x4xbf16>
}
// CHECK-LABEL:  func.func @reducemax_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x4xbf16>) -> tensor<1x?x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceMax"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x?x4xbf16>, tensor<1xi64>) -> tensor<1x?x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x?x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducemax_keepdims_zero(%arg0: tensor<1x1x4xbf16>) -> tensor<1x4xbf16> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<1xi64>) -> tensor<1x4xbf16>
  onnx.Return %0 : tensor<1x4xbf16>
}
// CHECK-LABEL:  func.func @reducemax_keepdims_zero
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 4]> : tensor<2xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1x4xbf16>, tensor<2xi64>) -> tensor<1x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducesum_full_tensor(%arg0: tensor<1x1x1024x1024xbf16>) -> tensor<1x1x1x1xbf16> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x1024x1024xbf16>, none) -> tensor<1x1x1x1xbf16>
  onnx.Return %0 : tensor<1x1x1x1xbf16>
}
// CHECK-LABEL:  func.func @reducesum_full_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1024x1024xbf16>) -> tensor<1x1x1x1xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x1024x1024xbf16>, tensor<2xi64>) -> tensor<1x1x1x1xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x1x1x1xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducesum_negative_axes(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
  %axes = onnx.Constant dense<[-3, -2]> : tensor<2xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<2xi64>) -> tensor<1x1x4xbf16>
  onnx.Return %0 : tensor<1x1x4xbf16>
}
// CHECK-LABEL:  func.func @reducesum_negative_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
// CHECK-NEXT:           onnx.Return [[PARAM_0_]] : tensor<1x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducesum_empty_default(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x1xbf16> {
  %axes = onnx.Constant dense<[]> : tensor<0xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<0xi64>) -> tensor<1x1x1xbf16>
  onnx.Return %0 : tensor<1x1x1xbf16>
}
// CHECK-LABEL:  func.func @reducesum_empty_default
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x1xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<1xi64>) -> tensor<1x1x1xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x1x1xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducesum_empty_noop(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
  %axes = onnx.Constant dense<[]> : tensor<0xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x1x4xbf16>, tensor<0xi64>) -> tensor<1x1x4xbf16>
  onnx.Return %0 : tensor<1x1x4xbf16>
}
// CHECK-LABEL:  func.func @reducesum_empty_noop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
// CHECK-NEXT:           onnx.Return [[PARAM_0_]] : tensor<1x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducesum_nonunit(%arg0: tensor<2x3x4xbf16>) -> tensor<2x1x4xbf16> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xbf16>, tensor<1xi64>) -> tensor<2x1x4xbf16>
  onnx.Return %0 : tensor<2x1x4xbf16>
}
// CHECK-LABEL:  func.func @reducesum_nonunit
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xbf16>) -> tensor<2x1x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xbf16>, tensor<1xi64>) -> tensor<2x1x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<2x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducesum_dynamic(%arg0: tensor<1x?x4xbf16>) -> tensor<1x?x4xbf16> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x?x4xbf16>, tensor<1xi64>) -> tensor<1x?x4xbf16>
  onnx.Return %0 : tensor<1x?x4xbf16>
}
// CHECK-LABEL:  func.func @reducesum_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x4xbf16>) -> tensor<1x?x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x?x4xbf16>, tensor<1xi64>) -> tensor<1x?x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x?x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reducesum_keepdims_zero(%arg0: tensor<1x1x4xbf16>) -> tensor<1x4xbf16> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<1xi64>) -> tensor<1x4xbf16>
  onnx.Return %0 : tensor<1x4xbf16>
}
// CHECK-LABEL:  func.func @reducesum_keepdims_zero
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 4]> : tensor<2xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1x4xbf16>, tensor<2xi64>) -> tensor<1x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reduceprod_full_tensor(%arg0: tensor<1x1x1024x1024xbf16>) -> tensor<1x1x1x1xbf16> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceProd"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x1024x1024xbf16>, none) -> tensor<1x1x1x1xbf16>
  onnx.Return %0 : tensor<1x1x1x1xbf16>
}
// CHECK-LABEL:  func.func @reduceprod_full_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1024x1024xbf16>) -> tensor<1x1x1x1xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceProd"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x1024x1024xbf16>, tensor<2xi64>) -> tensor<1x1x1x1xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x1x1x1xbf16>
// CHECK-NEXT:         }

// -----

func.func @reduceprod_negative_axes(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
  %axes = onnx.Constant dense<[-3, -2]> : tensor<2xi64>
  %0 = "onnx.ReduceProd"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<2xi64>) -> tensor<1x1x4xbf16>
  onnx.Return %0 : tensor<1x1x4xbf16>
}
// CHECK-LABEL:  func.func @reduceprod_negative_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
// CHECK-NEXT:           onnx.Return [[PARAM_0_]] : tensor<1x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reduceprod_empty_default(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x1xbf16> {
  %axes = onnx.Constant dense<[]> : tensor<0xi64>
  %0 = "onnx.ReduceProd"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<0xi64>) -> tensor<1x1x1xbf16>
  onnx.Return %0 : tensor<1x1x1xbf16>
}
// CHECK-LABEL:  func.func @reduceprod_empty_default
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x1xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceProd"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<1xi64>) -> tensor<1x1x1xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x1x1xbf16>
// CHECK-NEXT:         }

// -----

func.func @reduceprod_empty_noop(%arg0: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
  %axes = onnx.Constant dense<[]> : tensor<0xi64>
  %0 = "onnx.ReduceProd"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x1x4xbf16>, tensor<0xi64>) -> tensor<1x1x4xbf16>
  onnx.Return %0 : tensor<1x1x4xbf16>
}
// CHECK-LABEL:  func.func @reduceprod_empty_noop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x1x4xbf16> {
// CHECK-NEXT:           onnx.Return [[PARAM_0_]] : tensor<1x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reduceprod_nonunit(%arg0: tensor<2x3x4xbf16>) -> tensor<2x1x4xbf16> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.ReduceProd"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xbf16>, tensor<1xi64>) -> tensor<2x1x4xbf16>
  onnx.Return %0 : tensor<2x1x4xbf16>
}
// CHECK-LABEL:  func.func @reduceprod_nonunit
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xbf16>) -> tensor<2x1x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceProd"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xbf16>, tensor<1xi64>) -> tensor<2x1x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<2x1x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reduceprod_dynamic(%arg0: tensor<1x?x4xbf16>) -> tensor<1x?x4xbf16> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %0 = "onnx.ReduceProd"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x?x4xbf16>, tensor<1xi64>) -> tensor<1x?x4xbf16>
  onnx.Return %0 : tensor<1x?x4xbf16>
}
// CHECK-LABEL:  func.func @reduceprod_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x4xbf16>) -> tensor<1x?x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceProd"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x?x4xbf16>, tensor<1xi64>) -> tensor<1x?x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x?x4xbf16>
// CHECK-NEXT:         }

// -----

func.func @reduceprod_keepdims_zero(%arg0: tensor<1x1x4xbf16>) -> tensor<1x4xbf16> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %0 = "onnx.ReduceProd"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x4xbf16>, tensor<1xi64>) -> tensor<1x4xbf16>
  onnx.Return %0 : tensor<1x4xbf16>
}
// CHECK-LABEL:  func.func @reduceprod_keepdims_zero
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x4xbf16>) -> tensor<1x4xbf16> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 4]> : tensor<2xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1x4xbf16>, tensor<2xi64>) -> tensor<1x4xbf16>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1x4xbf16>
// CHECK-NEXT:         }

// -----

// A singleton reduction still performs the L1 elementwise transform.
func.func @reducel1_singleton(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %0 = "onnx.ReduceL1"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1xf32>, tensor<1xi64>) -> tensor<1xf32>
  onnx.Return %0 : tensor<1xf32>
}
// CHECK-LABEL:  func.func @reducel1_singleton
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1xf32>) -> tensor<1xf32> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceL1"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1xf32>
// CHECK-NEXT:         }

// -----

// A singleton reduction still performs the L2 elementwise transform.
func.func @reducel2_singleton(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %0 = "onnx.ReduceL2"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1xf32>, tensor<1xi64>) -> tensor<1xf32>
  onnx.Return %0 : tensor<1xf32>
}
// CHECK-LABEL:  func.func @reducel2_singleton
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1xf32>) -> tensor<1xf32> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1xf32>
// CHECK-NEXT:         }

// -----

// A singleton reduction still performs the SumSquare elementwise transform.
func.func @reducesumsquare_singleton(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %0 = "onnx.ReduceSumSquare"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1xf32>, tensor<1xi64>) -> tensor<1xf32>
  onnx.Return %0 : tensor<1xf32>
}
// CHECK-LABEL:  func.func @reducesumsquare_singleton
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1xf32>) -> tensor<1xf32> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceSumSquare"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1xf32>
// CHECK-NEXT:         }

// -----

// A singleton reduction still performs the LogSum elementwise transform.
func.func @reducelogsum_singleton(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %0 = "onnx.ReduceLogSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1xf32>, tensor<1xi64>) -> tensor<1xf32>
  onnx.Return %0 : tensor<1xf32>
}
// CHECK-LABEL:  func.func @reducelogsum_singleton
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1xf32>) -> tensor<1xf32> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceLogSum"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1xf32>
// CHECK-NEXT:         }

// -----

// A singleton reduction still performs the LogSumExp elementwise transform.
func.func @reducelogsumexp_singleton(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %0 = "onnx.ReduceLogSumExp"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1xf32>, tensor<1xi64>) -> tensor<1xf32>
  onnx.Return %0 : tensor<1xf32>
}
// CHECK-LABEL:  func.func @reducelogsumexp_singleton
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1xf32>) -> tensor<1xf32> {
// CHECK-NEXT:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT:           [[VAR_1_:%.+]] = "onnx.ReduceLogSumExp"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK-NEXT:           onnx.Return [[VAR_1_]] : tensor<1xf32>
// CHECK-NEXT:         }

