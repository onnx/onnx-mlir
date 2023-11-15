// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s

func.func @main_gather_elements(%arg0: tensor<3x2xf32>, %arg1: tensor<2x2xi64>) -> tensor<2x2xf32> {
  %0 = "onnx.GatherElements"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
// CHECK:  func.func @main_gather_elements([[PARAM_0_:%.+]]: tensor<3x2xf32>, [[PARAM_1_:%.+]]: tensor<2x2xi64>) -> tensor<2x2xf32> {
// CHECK-DAG:    [[CST_:%.+]] = arith.constant dense<[2, 2, 1]> : tensor<3xindex>
// CHECK-DAG:    [[VAR_0_:%.+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-DAG:    [[VAR_1_:%.+]] = stablehlo.constant dense<0> : tensor<2x2xi64>
// CHECK-DAG:    [[VAR_2_:%.+]] = stablehlo.broadcast_in_dim [[VAR_0_]], dims = [] : (tensor<i64>) -> tensor<2x2xi64>
// CHECK-DAG:    [[VAR_3_:%.+]] = stablehlo.compare  LT, [[PARAM_1_]], [[VAR_1_]],  NOTYPE : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi1>
// CHECK-DAG:    [[VAR_4_:%.+]] = stablehlo.add [[PARAM_1_]], [[VAR_2_]] : tensor<2x2xi64>
// CHECK-NEXT:   [[VAR_5_:%.+]] = stablehlo.select [[VAR_3_]], [[VAR_4_]], [[PARAM_1_]] : tensor<2x2xi1>, tensor<2x2xi64>
// CHECK-DAG:    [[VAR_6_:%.+]] = stablehlo.dynamic_reshape [[VAR_5_]], [[CST_]] : (tensor<2x2xi64>, tensor<3xindex>) -> tensor<2x2x1xi64>
// CHECK-DAG:    [[VAR_7_:%.+]] = stablehlo.dynamic_iota [[CST_]], dim = 1 : (tensor<3xindex>) -> tensor<2x2x1xi64>
// CHECK-NEXT:   [[VAR_8_:%.+]] = stablehlo.concatenate [[VAR_6_]], [[VAR_7_]], dim = 2 : (tensor<2x2x1xi64>, tensor<2x2x1xi64>) -> tensor<2x2x2xi64>
// CHECK-NEXT:   [[VAR_9_:%.+]] = "stablehlo.gather"([[PARAM_0_]], [[VAR_8_]]) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<3x2xf32>, tensor<2x2x2xi64>) -> tensor<2x2xf32>
// CHECK-NEXT:   return [[VAR_9_]] : tensor<2x2xf32>
}