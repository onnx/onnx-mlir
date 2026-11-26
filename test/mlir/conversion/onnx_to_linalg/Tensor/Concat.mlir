// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=onnx.Concat' %s -split-input-file | FileCheck %s

// -----

func.func @concat_axis_0(
    %arg0: tensor<2x3xf32>, %arg1: tensor<4x3xf32>) -> tensor<6x3xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64} :
      (tensor<2x3xf32>, tensor<4x3xf32>) -> tensor<6x3xf32>
  return %0 : tensor<6x3xf32>

  // CHECK-LABEL: func.func @concat_axis_0(
  // CHECK: [[RESULT:%.+]] = tensor.concat dim(0) %arg0, %arg1 : (tensor<2x3xf32>, tensor<4x3xf32>) -> tensor<6x3xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<6x3xf32>
}

// -----

func.func @concat_axis_1(
    %arg0: tensor<2x3xf32>, %arg1: tensor<2x5xf32>) -> tensor<2x8xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} :
      (tensor<2x3xf32>, tensor<2x5xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>

  // CHECK-LABEL: func.func @concat_axis_1(
  // CHECK: [[RESULT:%.+]] = tensor.concat dim(1) %arg0, %arg1 : (tensor<2x3xf32>, tensor<2x5xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<2x8xf32>
}

// -----

func.func @concat_negative_axis(
    %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x5xf32>)
    -> tensor<2x3x9xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = -1 : si64} :
      (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<2x3x9xf32>
  return %0 : tensor<2x3x9xf32>

  // CHECK-LABEL: func.func @concat_negative_axis(
  // CHECK: [[RESULT:%.+]] = tensor.concat dim(2) %arg0, %arg1 : (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<2x3x9xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<2x3x9xf32>
}

// -----

func.func @concat_three_inputs(%arg0: tensor<2xi64>, %arg1: tensor<3xi64>,
    %arg2: tensor<4xi64>) -> tensor<9xi64> {
  %0 = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 0 : si64} :
      (tensor<2xi64>, tensor<3xi64>, tensor<4xi64>) -> tensor<9xi64>
  return %0 : tensor<9xi64>

  // CHECK-LABEL: func.func @concat_three_inputs(
  // CHECK: [[RESULT:%.+]] = tensor.concat dim(0) %arg0, %arg1, %arg2 : (tensor<2xi64>, tensor<3xi64>, tensor<4xi64>) -> tensor<9xi64>
  // CHECK-NEXT: return [[RESULT]] : tensor<9xi64>
}

// -----

func.func @concat_dynamic_axis_dimension(
    %arg0: tensor<2x?xf32>, %arg1: tensor<2x?xf32>) -> tensor<2x?xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} :
      (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
  return %0 : tensor<2x?xf32>

  // CHECK-LABEL: func.func @concat_dynamic_axis_dimension(
  // CHECK: [[RESULT:%.+]] = tensor.concat dim(1) %arg0, %arg1 : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<2x?xf32>
}

// -----

func.func @concat_dynamic_non_axis_dimension(
    %arg0: tensor<?x3xf32>, %arg1: tensor<?x5xf32>) -> tensor<?x8xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} :
      (tensor<?x3xf32>, tensor<?x5xf32>) -> tensor<?x8xf32>
  return %0 : tensor<?x8xf32>

  // CHECK-LABEL: func.func @concat_dynamic_non_axis_dimension(
  // CHECK: [[RESULT:%.+]] = tensor.concat dim(1) %arg0, %arg1 : (tensor<?x3xf32>, tensor<?x5xf32>) -> tensor<?x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<?x8xf32>
}

// -----

func.func @concat_dynamic_inputs_static_result(
    %arg0: tensor<2x?xf32>, %arg1: tensor<2x?xf32>) -> tensor<2x8xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} :
      (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>

// CHECK-LABEL:  func.func @concat_dynamic_inputs_static_result
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x?xf32>, [[PARAM_1_:%.+]]: tensor<2x?xf32>) -> tensor<2x8xf32> {
// CHECK:           [[VAR_concat_:%.+]] = tensor.concat dim(1) [[PARAM_0_]], [[PARAM_1_]] : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x8xf32>
// CHECK:           return [[VAR_concat_]] : tensor<2x8xf32>
// CHECK:         }

}

// -----

func.func @concat_single_input(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Concat"(%arg0) {axis = 0 : si64} :
      (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>

  // CHECK-LABEL: func.func @concat_single_input(
  // CHECK-NOT: "onnx.Concat"
  // CHECK-NOT: tensor.concat
  // CHECK: return %arg0 : tensor<2x3xf32>
}

// -----

func.func @concat_zero_extent(
    %arg0: tensor<2x0xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} :
      (tensor<2x0xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>

  // CHECK-LABEL: func.func @concat_zero_extent(
  // CHECK: [[RESULT:%.+]] = tensor.concat dim(1) %arg0, %arg1 : (tensor<2x0xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<2x3xf32>
}

// -----

func.func @concat_ignores_rank_one_empty(%empty: tensor<0xf32>,
    %arg0: tensor<2x3xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x7xf32> {
  %0 = "onnx.Concat"(%empty, %arg0, %arg1) {axis = -1 : si64} :
      (tensor<0xf32>, tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  return %0 : tensor<2x7xf32>

  // CHECK-LABEL: func.func @concat_ignores_rank_one_empty(
  // CHECK-SAME: [[EMPTY:%.*]]: tensor<0xf32>, [[ARG0:%.*]]: tensor<2x3xf32>, [[ARG1:%.*]]: tensor<2x4xf32>
  // CHECK-NOT: tensor.concat {{.*}}[[EMPTY]]
  // CHECK: [[RESULT:%.+]] = tensor.concat dim(1) [[ARG0]], [[ARG1]] : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<2x7xf32>
}

// -----

func.func @concat_unranked_not_lowered(
    %arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64} :
      (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func.func @concat_unranked_not_lowered(
  // CHECK-NOT: tensor.concat
  // CHECK: [[RESULT:%.+]] = "onnx.Concat"(%arg0, %arg1)
  // CHECK: return [[RESULT]] : tensor<*xf32>
}
