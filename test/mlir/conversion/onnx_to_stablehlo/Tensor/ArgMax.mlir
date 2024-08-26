// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo %s --canonicalize -cse -split-input-file | FileCheck %s

func.func @test_argmax_verifier_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xi64> {
  %1 = "onnx.ArgMax"(%arg0) { axis = -1 : si64} : (tensor<5x5x1x32xf32>)  -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
}

// CHECK-LABEL:  func.func @test_argmax_verifier_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<5x5x1x1xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [5, 5, 1, 32] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.dynamic_iota [[VAR_0_]], dim = 3 : (tensor<4xindex>) -> tensor<5x5x1x32xi64>
// CHECK:           [[VAR_4_:%.+]]:2 = stablehlo.reduce(%arg0 init: [[VAR_2_]]), (%1 init: [[VAR_1_]]) across dimensions = [3] : (tensor<5x5x1x32xf32>, tensor<5x5x1x32xi64>, tensor<f32>, tensor<i64>) -> (tensor<5x5x1xf32>, tensor<5x5x1xi64>)
// CHECK:            reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
// CHECK:             [[VAR_6_:%.+]] = stablehlo.compare  GE, %arg1, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-DAG:         [[VAR_7_:%.+]] = stablehlo.select [[VAR_6_]], %arg1, %arg3 : tensor<i1>, tensor<f32>
// CHECK-DAG:         [[VAR_8_:%.+]] = stablehlo.compare  EQ, %arg1, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-DAG:         [[VAR_9_:%.+]] = stablehlo.minimum %arg2, %arg4 : tensor<i64>
// CHECK-DAG:         [[VAR_10_:%.+]] = stablehlo.select [[VAR_6_]], %arg2, %arg4 : tensor<i1>, tensor<i64>
// CHECK:             [[VAR_11_:%.+]] = stablehlo.select [[VAR_8_]], [[VAR_9_]], [[VAR_10_]] : tensor<i1>, tensor<i64>
// CHECK:             stablehlo.return [[VAR_7_]], [[VAR_11_]] : tensor<f32>, tensor<i64>
// CHECK:           }
// CHECK:           [[VAR_5_:%.+]] = stablehlo.reshape [[VAR_4_]]#1 : (tensor<5x5x1xi64>) -> tensor<5x5x1x1xi64>
// CHECK:           return [[VAR_5_]] : tensor<5x5x1x1xi64>
// CHECK:         }

// -----

func.func @test_argmax_verifier_2(%arg0 : tensor<5x?x1x32xf32>) -> tensor<*xi64> {
  %1 = "onnx.ArgMax"(%arg0) { axis = 3 : si64} : (tensor<5x?x1x32xf32>)  -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
}

// CHECK-LABEL:  func.func @test_argmax_verifier_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x?x1x32xf32>) -> tensor<5x?x1x1xi64> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<5x?x1x32xf32> -> tensor<4xindex>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.dynamic_iota [[VAR_2_]], dim = 3 : (tensor<4xindex>) -> tensor<5x?x1x32xi64>
// CHECK:           [[VAR_4_:%.+]]:2 = stablehlo.reduce([[PARAM_0_]] init: [[VAR_0_]]), ([[VAR_3_]] init: [[VAR_1_]]) across dimensions = [3] : (tensor<5x?x1x32xf32>, tensor<5x?x1x32xi64>, tensor<f32>, tensor<i64>) -> (tensor<5x?x1xf32>, tensor<5x?x1xi64>)
// CHECK:            reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
// CHECK:             [[VAR_11_:%.+]] = stablehlo.compare  GE, %arg1, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-DAG:         [[VAR_12_:%.+]] = stablehlo.select [[VAR_11_]], %arg1, %arg3 : tensor<i1>, tensor<f32>
// CHECK-DAG:         [[VAR_13_:%.+]] = stablehlo.compare  EQ, %arg1, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-DAG:         [[VAR_14_:%.+]] = stablehlo.minimum %arg2, %arg4 : tensor<i64>
// CHECK-DAG:         [[VAR_15_:%.+]] = stablehlo.select [[VAR_11_]], %arg2, %arg4 : tensor<i1>, tensor<i64>
// CHECK:             [[VAR_16_:%.+]] = stablehlo.select [[VAR_13_]], [[VAR_14_]], [[VAR_15_]] : tensor<i1>, tensor<i64>
// CHECK:             stablehlo.return [[VAR_12_]], [[VAR_16_]] : tensor<f32>, tensor<i64>
// CHECK:           }
// CHECK-DAG:       [[VAR_5_:%.+]] = shape.get_extent [[VAR_2_]], [[CST_0_]] : tensor<4xindex>, index -> index
// CHECK-DAG:       [[VAR_6_:%.+]] = shape.get_extent [[VAR_2_]], [[CST_1_]] : tensor<4xindex>, index -> index
// CHECK-DAG:       [[VAR_7_:%.+]] = shape.get_extent [[VAR_2_]], [[CST_2_]] : tensor<4xindex>, index -> index
// CHECK:           [[VAR_8_:%.+]] = shape.from_extents [[VAR_5_]], [[VAR_6_]], [[VAR_7_]], [[CST_1_]] : index, index, index, index
// CHECK:           [[VAR_9_:%.+]] = shape.to_extent_tensor [[VAR_8_]] : !shape.shape -> tensor<4xindex>
// CHECK:           [[VAR_10_:%.+]] = stablehlo.dynamic_reshape [[VAR_4_]]#1, [[VAR_9_]] : (tensor<5x?x1xi64>, tensor<4xindex>) -> tensor<5x?x1x1xi64>
// CHECK:           return [[VAR_10_]] : tensor<5x?x1x1xi64>
// CHECK:         }
