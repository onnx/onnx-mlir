// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_onehot(%arg0 : tensor<2x3x4xi64>) -> tensor<*xi64> {
  %0 = onnx.Constant dense<64> : tensor<1xi64>
  %1 = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %2 = "onnx.OneHot"(%arg0, %0, %1) {axis = -1 : si64} : (tensor<2x3x4xi64>, tensor<1xi64>, tensor<2xi64>) -> tensor<*xi64>
  "func.return"(%2) : (tensor<*xi64>) -> ()
// CHECK-LABEL:  func.func @test_onehot
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xi64>) -> tensor<2x3x4x64xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<64> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.iota dim = 3 : tensor<2x3x4x64xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_0_]], dims = [0, 1, 2] : (tensor<2x3x4xi64>) -> tensor<2x3x4x64xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.broadcast_in_dim [[VAR_0_]], dims = [] : (tensor<i64>) -> tensor<2x3x4x64xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.broadcast_in_dim [[VAR_1_]], dims = [0] : (tensor<1xi64>) -> tensor<2x3x4x64xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = stablehlo.compare  GE, [[VAR_4_]], [[VAR_5_]] : (tensor<2x3x4x64xi64>, tensor<2x3x4x64xi64>) -> tensor<2x3x4x64xi1>
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.add [[VAR_4_]], [[VAR_6_]] : tensor<2x3x4x64xi64>
// CHECK:           [[VAR_9_:%.+]] = stablehlo.select [[VAR_7_]], [[VAR_4_]], [[VAR_8_]] : tensor<2x3x4x64xi1>, tensor<2x3x4x64xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.compare  EQ, [[VAR_9_]], [[VAR_3_]] : (tensor<2x3x4x64xi64>, tensor<2x3x4x64xi64>) -> tensor<2x3x4x64xi1>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.slice [[VAR_2_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.slice [[VAR_2_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.broadcast_in_dim [[VAR_11_]], dims = [0] : (tensor<1xi64>) -> tensor<2x3x4x64xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.broadcast_in_dim [[VAR_12_]], dims = [0] : (tensor<1xi64>) -> tensor<2x3x4x64xi64>
// CHECK:           [[VAR_15_:%.+]] = stablehlo.select [[VAR_10_]], [[VAR_14_]], [[VAR_13_]] : tensor<2x3x4x64xi1>, tensor<2x3x4x64xi64>
// CHECK:           return [[VAR_15_]] : tensor<2x3x4x64xi64>
// CHECK:         }
}
