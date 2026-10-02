// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --enable-keepdims-canonicalization=true --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --enable-keepdims-canonicalization=false --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s -split-input-file | FileCheck %s --check-prefix=DISABLED-CHECK


func.func @test_reducesumv11_axes(%arg0: tensor<1x32x512x640xf32>) -> tensor<1x512x640xf32> {
  %0 = "onnx.ReduceSumV11"(%arg0) {axes = [1], keepdims = 0 : si64} : (tensor<1x32x512x640xf32>) -> tensor<1x512x640xf32>
  onnx.Return %0 : tensor<1x512x640xf32>
// CHECK-LABEL:  func.func @test_reducesumv11_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x512x640xf32>) -> tensor<1x512x640xf32> {
// CHECK-DAG:       [[SHAPE_:%.+]] = onnx.Constant dense<[1, 512, 640]> : tensor<3xi64>
// CHECK-DAG:       [[AXES_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[REDUCED_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[AXES_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x32x512x640xf32>, tensor<1xi64>) -> tensor<1x1x512x640xf32>
// CHECK:           [[RES_:%.+]] = "onnx.Reshape"([[REDUCED_]], [[SHAPE_]]) {allowzero = 0 : si64} : (tensor<1x1x512x640xf32>, tensor<3xi64>) -> tensor<1x512x640xf32>
// CHECK:           onnx.Return [[RES_]] : tensor<1x512x640xf32>
// CHECK:         }
// DISABLED-CHECK-LABEL:  func.func @test_reducesumv11_axes
// DISABLED-CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x512x640xf32>) -> tensor<1x512x640xf32> {
// DISABLED-CHECK:           [[AXES_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// DISABLED-CHECK:           [[REDUCED_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[AXES_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x32x512x640xf32>, tensor<1xi64>) -> tensor<1x512x640xf32>
// DISABLED-CHECK:           onnx.Return [[REDUCED_]] : tensor<1x512x640xf32>
// DISABLED-CHECK-NOT:       "onnx.Reshape"
}

// -----

func.func @test_reducesumv11_noaxes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %0 = "onnx.ReduceSumV11"(%arg0) {keepdims = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL:  func.func @test_reducesumv11_noaxes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<1x1x1xf32>
// CHECK:         }
}

// -----

// Rewrite keepdims=0 to keepdims=1 + Reshape for ONNX reduce ops.

// CHECK-LABEL: func.func @reduce_l2_keepdims_zero
func.func @reduce_l2_keepdims_zero(%arg0: tensor<2x3x4xf32>) -> tensor<2x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.ReduceL2"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
  onnx.Return %0 : tensor<2x4xf32>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  // CHECK-DAG: [[AXES:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: [[REDUCE:%.+]] = "onnx.ReduceL2"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[REDUCE]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<2x1x4xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @reduce_l2_keepdims_zero
  // DISABLED-CHECK: "onnx.ReduceL2"(%arg0, %{{.*}}) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @reduce_max_keepdims_zero
func.func @reduce_max_keepdims_zero(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3xf32> {
  %axes = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<1x3x5x5xf32>, tensor<2xi64>) -> tensor<1x3xf32>
  onnx.Return %0 : tensor<1x3xf32>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  // CHECK-DAG: [[AXES:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  // CHECK: [[REDUCE:%.+]] = "onnx.ReduceMax"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<1x3x5x5xf32>, tensor<2xi64>) -> tensor<1x3x1x1xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[REDUCE]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<1x3x1x1xf32>, tensor<2xi64>) -> tensor<1x3xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @reduce_max_keepdims_zero
  // DISABLED-CHECK: "onnx.ReduceMax"(%arg0, %{{.*}}) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @reduce_mean_keepdims_zero
func.func @reduce_mean_keepdims_zero(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x4xf32> {
  %axes = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4x5xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  onnx.Return %0 : tensor<2x4xf32>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  // CHECK-DAG: [[AXES:%.+]] = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  // CHECK: [[REDUCE:%.+]] = "onnx.ReduceMean"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4x5xf32>, tensor<2xi64>) -> tensor<2x1x4x1xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[REDUCE]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<2x1x4x1xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @reduce_mean_keepdims_zero
  // DISABLED-CHECK: "onnx.ReduceMean"(%arg0, %{{.*}}) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @reduce_min_keepdims_zero
func.func @reduce_min_keepdims_zero(%arg0: tensor<2x3x4xf32>) -> tensor<3xf32> {
  %axes = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  %0 = "onnx.ReduceMin"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
  // CHECK-DAG: [[AXES:%.+]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  // CHECK: [[REDUCE:%.+]] = "onnx.ReduceMin"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<1x3x1xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[REDUCE]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<1x3x1xf32>, tensor<1xi64>) -> tensor<3xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @reduce_min_keepdims_zero
  // DISABLED-CHECK: "onnx.ReduceMin"(%arg0, %{{.*}}) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @reduce_sum_keepdims_zero
func.func @reduce_sum_keepdims_zero(%arg0: tensor<1x2x3x4xf32>) -> tensor<2x3xf32> {
  %axes = onnx.Constant dense<[0, 3]> : tensor<2xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<1x2x3x4xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  onnx.Return %0 : tensor<2x3xf32>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  // CHECK-DAG: [[AXES:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
  // CHECK: [[REDUCE:%.+]] = "onnx.ReduceSum"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<1x2x3x4xf32>, tensor<1xi64>) -> tensor<1x2x3x1xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[REDUCE]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<1x2x3x1xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @reduce_sum_keepdims_zero
  // DISABLED-CHECK: "onnx.ReduceSum"(%arg0, %{{.*}}) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @reduce_l1_keepdims_zero
func.func @reduce_l1_keepdims_zero(%arg0: tensor<2x3x4xf32>) -> tensor<2x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.ReduceL1"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
  onnx.Return %0 : tensor<2x4xf32>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  // CHECK-DAG: [[AXES:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: [[REDUCE:%.+]] = "onnx.ReduceL1"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[REDUCE]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<2x1x4xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @reduce_l1_keepdims_zero
  // DISABLED-CHECK: "onnx.ReduceL1"(%arg0, %{{.*}}) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @reduce_log_sum_keepdims_zero
func.func @reduce_log_sum_keepdims_zero(%arg0: tensor<2x3x4xf32>) -> tensor<2x3xf32> {
  %axes = onnx.Constant dense<[2]> : tensor<1xi64>
  %0 = "onnx.ReduceLogSum"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x3xf32>
  onnx.Return %0 : tensor<2x3xf32>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  // CHECK-DAG: [[AXES:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
  // CHECK: [[REDUCE:%.+]] = "onnx.ReduceLogSum"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x3x1xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[REDUCE]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<2x3x1xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @reduce_log_sum_keepdims_zero
  // DISABLED-CHECK: "onnx.ReduceLogSum"(%arg0, %{{.*}}) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @reduce_log_sum_exp_keepdims_zero
func.func @reduce_log_sum_exp_keepdims_zero(%arg0: tensor<1x2x3x4xf32>) -> tensor<2x3xf32> {
  %axes = onnx.Constant dense<[0, 3]> : tensor<2xi64>
  %0 = "onnx.ReduceLogSumExp"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<1x2x3x4xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  onnx.Return %0 : tensor<2x3xf32>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  // CHECK-DAG: [[AXES:%.+]] = onnx.Constant dense<[0, 3]> : tensor<2xi64>
  // CHECK: [[REDUCE:%.+]] = "onnx.ReduceLogSumExp"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<1x2x3x4xf32>, tensor<2xi64>) -> tensor<1x2x3x1xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[REDUCE]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<1x2x3x1xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @reduce_log_sum_exp_keepdims_zero
  // DISABLED-CHECK: "onnx.ReduceLogSumExp"(%arg0, %{{.*}}) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @reduce_prod_keepdims_zero
func.func @reduce_prod_keepdims_zero(%arg0: tensor<2x3x4xf32>) -> tensor<3xf32> {
  %axes = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  %0 = "onnx.ReduceProd"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
  // CHECK-DAG: [[AXES:%.+]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  // CHECK: [[REDUCE:%.+]] = "onnx.ReduceProd"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<1x3x1xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[REDUCE]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<1x3x1xf32>, tensor<1xi64>) -> tensor<3xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @reduce_prod_keepdims_zero
  // DISABLED-CHECK: "onnx.ReduceProd"(%arg0, %{{.*}}) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @reduce_sum_square_keepdims_zero
func.func @reduce_sum_square_keepdims_zero(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x4xf32> {
  %axes = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %0 = "onnx.ReduceSumSquare"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4x5xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  onnx.Return %0 : tensor<2x4xf32>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  // CHECK-DAG: [[AXES:%.+]] = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  // CHECK: [[REDUCE:%.+]] = "onnx.ReduceSumSquare"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4x5xf32>, tensor<2xi64>) -> tensor<2x1x4x1xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[REDUCE]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<2x1x4x1xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @reduce_sum_square_keepdims_zero
  // DISABLED-CHECK: "onnx.ReduceSumSquare"(%arg0, %{{.*}}) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @reduce_sum_keepdims_one_unchanged
func.func @reduce_sum_keepdims_one_unchanged(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
  onnx.Return %0 : tensor<2x1x4xf32>
  // CHECK: "onnx.ReduceSum"(%arg0, %{{.*}}) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-NOT: "onnx.Reshape"
  // DISABLED-CHECK-LABEL: func.func @reduce_sum_keepdims_one_unchanged
  // DISABLED-CHECK: "onnx.ReduceSum"(%arg0, %{{.*}}) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @argmax_keepdims_zero
func.func @argmax_keepdims_zero(%arg0: tensor<2x3x4xf32>) -> tensor<2x4xi64> {
  %0 = "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 0 : si64,
      select_last_index = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<2x4xi64>
  onnx.Return %0 : tensor<2x4xi64>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  // CHECK: [[ARGMAX:%.+]] = "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 1 : si64, select_last_index = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4xf32>) -> tensor<2x1x4xi64>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[ARGMAX]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<2x1x4xi64>, tensor<2xi64>) -> tensor<2x4xi64>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @argmax_keepdims_zero
  // DISABLED-CHECK: "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 0 : si64, select_last_index = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @argmin_keepdims_zero
func.func @argmin_keepdims_zero(%arg0: tensor<2x3x4xf32>) -> tensor<2x4xi64> {
  %0 = "onnx.ArgMin"(%arg0) {axis = 1 : si64, keepdims = 0 : si64,
      select_last_index = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<2x4xi64>
  onnx.Return %0 : tensor<2x4xi64>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  // CHECK: [[ARGMIN:%.+]] = "onnx.ArgMin"(%arg0) {axis = 1 : si64, keepdims = 1 : si64, select_last_index = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4xf32>) -> tensor<2x1x4xi64>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[ARGMIN]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<2x1x4xi64>, tensor<2xi64>) -> tensor<2x4xi64>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @argmin_keepdims_zero
  // DISABLED-CHECK: "onnx.ArgMin"(%arg0) {axis = 1 : si64, keepdims = 0 : si64, select_last_index = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// CHECK-LABEL: func.func @argmax_negative_axis_keepdims_zero
func.func @argmax_negative_axis_keepdims_zero(%arg0: tensor<2x3x4xf32>) -> tensor<2x3xi64> {
  %0 = "onnx.ArgMax"(%arg0) {axis = -1 : si64, keepdims = 0 : si64,
      select_last_index = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<2x3xi64>
  onnx.Return %0 : tensor<2x3xi64>
  // CHECK-DAG: [[SHAPE:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  // CHECK: [[ARGMAX:%.+]] = "onnx.ArgMax"(%arg0) {axis = 2 : si64, keepdims = 1 : si64, select_last_index = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4xf32>) -> tensor<2x3x1xi64>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[ARGMAX]], [[SHAPE]]) {allowzero = 0 : si64}
  // CHECK-SAME: (tensor<2x3x1xi64>, tensor<2xi64>) -> tensor<2x3xi64>
  // CHECK: onnx.Return [[RES]]
  // CHECK-NOT: keepdims = 0
  // DISABLED-CHECK-LABEL: func.func @argmax_negative_axis_keepdims_zero
  // DISABLED-CHECK: "onnx.ArgMax"(%arg0) {axis = 2 : si64, keepdims = 0 : si64, select_last_index = 0 : si64}
  // DISABLED-CHECK: onnx.Return %{{.*}}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
}

// -----
// A dynamic result shape prevents the keepdims canonicalization.
func.func @argmax_dynamic_result_shape(%arg0: tensor<?x3x4xf32>) -> tensor<?x4xi64> {
  %0 = "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 0 : si64,
      select_last_index = 0 : si64} : (tensor<?x3x4xf32>) -> tensor<?x4xi64>
  onnx.Return %0 : tensor<?x4xi64>
  // CHECK-LABEL: func.func @argmax_dynamic_result_shape
  // CHECK: "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 0 : si64, select_last_index = 0 : si64}
  // CHECK-NOT: "onnx.Reshape"
  // CHECK: onnx.Return %{{.*}} : tensor<?x4xi64>
  // DISABLED-CHECK-LABEL: func.func @argmax_dynamic_result_shape
  // DISABLED-CHECK: "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 0 : si64, select_last_index = 0 : si64}
  // DISABLED-CHECK-NOT: "onnx.Reshape"
  // DISABLED-CHECK: onnx.Return %{{.*}} : tensor<?x4xi64>
}
