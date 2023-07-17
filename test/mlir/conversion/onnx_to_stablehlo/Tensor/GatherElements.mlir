// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s

func.func @main_gather_elements(%arg0: tensor<3x2xf32>, %arg1: tensor<2x2xi64>) -> tensor<2x2xf32> {
  %0 = "onnx.GatherElements"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
// CHECK:  func.func @main_gather_elements([[PARAM_0_:%.+]]: tensor<3x2xf32>, [[PARAM_1_:%.+]]: tensor<2x2xi64>) -> tensor<2x2xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = stablehlo.constant dense<3> : tensor<2x2xi64>
// CHECK-DAG:    [[VAR_1_:%.+]] = stablehlo.constant dense<0> : tensor<2x2xi64>
// CHECK-DAG:    [[VAR_2_:%.+]] = stablehlo.compare  LT, [[PARAM_1_]], [[VAR_1_]],  NOTYPE : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi1>
// CHECK-DAG:    [[VAR_3_:%.+]] = stablehlo.add [[PARAM_1_]], [[VAR_0_]] : tensor<2x2xi64>
// CHECK-NEXT:   [[VAR_4_:%.+]] = stablehlo.select [[VAR_2_]], [[VAR_3_]], [[PARAM_1_]] : tensor<2x2xi1>, tensor<2x2xi64>
// CHECK-NEXT:   [[VAR_5_:%.+]] = stablehlo.reshape [[VAR_4_]] : (tensor<2x2xi64>) -> tensor<2x2x1xi64>
// CHECK-DAG:    [[VAR_6_:%.+]] = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<2xi64>
// CHECK-NEXT:   [[VAR_7_:%.+]] = "stablehlo.broadcast_in_dim"([[VAR_6_]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<2x2x1xi64>
// CHECK-NEXT:   [[VAR_8_:%.+]] = "stablehlo.concatenate"([[VAR_5_]], [[VAR_7_]]) {dimension = 2 : i64} : (tensor<2x2x1xi64>, tensor<2x2x1xi64>) -> tensor<2x2x2xi64>
// CHECK-NEXT:   [[VAR_9_:%.+]] = "stablehlo.gather"([[PARAM_0_]], [[VAR_8_]]) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<3x2xf32>, tensor<2x2x2xi64>) -> tensor<2x2xf32>
// CHECK-NEXT:   return [[VAR_9_]] : tensor<2x2xf32>
}