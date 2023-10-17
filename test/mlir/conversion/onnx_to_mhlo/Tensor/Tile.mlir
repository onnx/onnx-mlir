// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

// Test tile with constant repeats
func.func @test_tile1(%arg0 : tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() { value = dense<[3, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Tile"(%arg0, %0) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

// CHECK-LABEL:  func.func @test_tile1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>) -> tensor<12x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<8> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.slice [[VAR_0_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.slice [[VAR_0_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.concatenate [[VAR_3_]], [[VAR_1_]], [[VAR_4_]], [[VAR_2_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_5_]], dims = [1, 3] : (tensor<4x8xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = stablehlo.multiply [[VAR_3_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.multiply [[VAR_4_]], [[VAR_2_]] : tensor<1xi64>
// CHECK:           [[VAR_9_:%.+]] = stablehlo.concatenate [[VAR_7_]], [[VAR_8_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_10_:%.+]] = stablehlo.dynamic_reshape [[VAR_6_]], [[VAR_9_]] : (tensor<?x?x?x?xf32>, tensor<2xi64>) -> tensor<12x16xf32>
// CHECK:           return [[VAR_10_]] : tensor<12x16xf32>
// CHECK:         }

// -----

func.func @test_tile_dynamic(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Tile"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_tile_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>, [[PARAM_1_:%.+]]: tensor<4xi64>) -> tensor<?x?x?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<5> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.slice [[PARAM_1_]] [0:1] : (tensor<4xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.slice [[PARAM_1_]] [1:2] : (tensor<4xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.slice [[PARAM_1_]] [2:3] : (tensor<4xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.slice [[PARAM_1_]] [3:4] : (tensor<4xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.concatenate [[VAR_3_]], [[VAR_0_]], [[VAR_4_]], [[VAR_0_]], [[VAR_5_]], [[VAR_1_]], [[VAR_6_]], [[VAR_2_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<8xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_7_]], dims = [1, 3, 5, 7] : (tensor<5x5x1x32xf32>, tensor<8xi64>) -> tensor<?x?x?x?x?x?x?x?xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.multiply [[VAR_3_]], [[VAR_0_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.multiply [[VAR_4_]], [[VAR_0_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.multiply [[VAR_5_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.multiply [[VAR_6_]], [[VAR_2_]] : tensor<1xi64>
// CHECK:           [[VAR_13_:%.+]] = stablehlo.concatenate [[VAR_9_]], [[VAR_10_]], [[VAR_11_]], [[VAR_12_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_14_:%.+]] = stablehlo.dynamic_reshape [[VAR_8_]], [[VAR_13_]] : (tensor<?x?x?x?x?x?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
// CHECK:           return [[VAR_14_]] : tensor<?x?x?x?xf32>
// CHECK:         }
