// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s

func.func @main_gather_elements(%arg0: tensor<3x2xf32>, %arg1: tensor<2x2xi64>) -> tensor<2x2xf32> {
  %0 = "onnx.GatherElements"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @main_gather_elements
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2xf32>, [[PARAM_1_:%.+]]: tensor<2x2xi64>) -> tensor<2x2xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<[2, 2, 1]> : tensor<3xindex>
// CHECK-DAG:       [[VAR_c_:%.+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-DAG:       [[VAR_c_0_:%.+]] = stablehlo.constant dense<0> : tensor<2x2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.broadcast_in_dim [[VAR_c_]], dims = [] : (tensor<i64>) -> tensor<2x2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.compare  LT, [[PARAM_1_]], [[VAR_c_0_]] : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi1>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.add [[PARAM_1_]], [[VAR_0_]] : tensor<2x2xi64>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.select [[VAR_1_]], [[VAR_2_]], [[PARAM_1_]] : tensor<2x2xi1>, tensor<2x2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.dynamic_reshape [[VAR_3_]], [[VAR_cst_]] : (tensor<2x2xi64>, tensor<3xindex>) -> tensor<2x2x1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.dynamic_iota [[VAR_cst_]], dim = 1 : (tensor<3xindex>) -> tensor<2x2x1xi64>
// CHECK:           [[VAR_6_:%.+]] = stablehlo.concatenate [[VAR_4_]], [[VAR_5_]], dim = 2 : (tensor<2x2x1xi64>, tensor<2x2x1xi64>) -> tensor<2x2x2xi64>
// CHECK:           [[VAR_7_:%.+]] = "stablehlo.gather"([[PARAM_0_]], [[VAR_6_]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<3x2xf32>, tensor<2x2x2xi64>) -> tensor<2x2xf32>
// CHECK:           return [[VAR_7_]] : tensor<2x2xf32>
// CHECK:         }
}