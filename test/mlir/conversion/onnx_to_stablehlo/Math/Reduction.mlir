// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_reducemax_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_reducemax_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>) -> tensor<3x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_0_]]) applies stablehlo.maximum across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK:           return [[VAR_1_]] : tensor<3x2xf32>
// CHECK:         }

// -----

func.func @test_reducemax_v13_keepdims(%arg0: tensor<?x20x30xf32>) -> tensor<?x1x30xf32> {
  %0 = "onnx.ReduceMaxV13"(%arg0) {axes = [1], keepdims = 1 : si64} : (tensor<?x20x30xf32>) -> tensor<?x1x30xf32>
  return %0 : tensor<?x1x30xf32>
}

// CHECK-LABEL:  func.func @test_reducemax_v13_keepdims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20x30xf32>) -> tensor<?x1x30xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_0_]]) applies stablehlo.maximum across dimensions = [1] : (tensor<?x20x30xf32>, tensor<f32>) -> tensor<?x30xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.get_extent [[VAR_2_]], [[CST_0_]] : tensor<3xindex>, index -> index
// CHECK-DAG:       [[VAR_4_:%.+]] = shape.get_extent [[VAR_2_]], [[CST_2_]] : tensor<3xindex>, index -> index
// CHECK:           [[VAR_5_:%.+]] = shape.from_extents [[VAR_3_]], [[CST_1_]], [[VAR_4_]] : index, index, index
// CHECK:           [[VAR_6_:%.+]] = shape.to_extent_tensor [[VAR_5_]] : !shape.shape -> tensor<3xindex>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.dynamic_reshape [[VAR_1_]], [[VAR_6_]] : (tensor<?x30xf32>, tensor<3xindex>) -> tensor<?x1x30xf32>
// CHECK:           return [[VAR_7_]] : tensor<?x1x30xf32>
// CHECK:         }

// -----

func.func @test_reducemax_v13_integer_tensor(%arg0 : tensor<3x2x2xi64>) -> tensor<3x2xi64> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xi64>)-> tensor<3x2xi64>
  "func.return"(%0) : (tensor<3x2xi64>) -> ()
}

// CHECK-LABEL:  func.func @test_reducemax_v13_integer_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xi64>) -> tensor<3x2xi64> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.constant dense<-9223372036854775808> : tensor<i64>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_0_]]) applies stablehlo.maximum across dimensions = [1] : (tensor<3x2x2xi64>, tensor<i64>) -> tensor<3x2xi64>
// CHECK:           return [[VAR_1_]] : tensor<3x2xi64>
// CHECK:         }

// -----

func.func @test_reducemin_v13(%arg0 : tensor<?x2x2xf32>) -> tensor<?x2xf32> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x2x2xf32>)-> tensor<?x2xf32>
  "func.return"(%0) : (tensor<?x2xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_reducemin_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x2x2xf32>) -> tensor<?x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_0_]]) applies stablehlo.minimum across dimensions = [1] : (tensor<?x2x2xf32>, tensor<f32>) -> tensor<?x2xf32>
// CHECK:           return [[VAR_1_]] : tensor<?x2xf32>
// CHECK:         }

// -----

func.func @test_reducemin_v13_integer_tensor(%arg0 : tensor<?x2x2xi64>) -> tensor<?x2xi64> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x2x2xi64>)-> tensor<?x2xi64>
  "func.return"(%0) : (tensor<?x2xi64>) -> ()
}

// CHECK-LABEL:  func.func @test_reducemin_v13_integer_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x2x2xi64>) -> tensor<?x2xi64> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.constant dense<9223372036854775807> : tensor<i64>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_0_]]) applies stablehlo.minimum across dimensions = [1] : (tensor<?x2x2xi64>, tensor<i64>) -> tensor<?x2xi64>
// CHECK:           return [[VAR_1_]] : tensor<?x2xi64>
// CHECK:         }

// -----

func.func @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %cst = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 ="onnx.ReduceSum"(%arg0, %cst) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_reducesum
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>) -> tensor<3x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_0_]]) applies stablehlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK:           return [[VAR_1_]] : tensor<3x2xf32>
// CHECK:         }

// -----

func.func @test_reducesum_v11(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %0 ="onnx.ReduceSumV11"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_reducesum_v11
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>) -> tensor<3x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_0_]]) applies stablehlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK:           return [[VAR_1_]] : tensor<3x2xf32>
// CHECK:         }

// -----

func.func @test_reducesum1(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
  return %0 : tensor<3x1x2xf32>
}

// CHECK-LABEL:  func.func @test_reducesum1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>, [[PARAM_1_:%.+]]: tensor<?xi64>) -> tensor<3x1x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [3, 1, 2] : tensor<3xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies stablehlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.dynamic_reshape [[VAR_2_]], [[VAR_0_]] : (tensor<3x2xf32>, tensor<3xindex>) -> tensor<3x1x2xf32>
// CHECK:           return [[VAR_3_]] : tensor<3x1x2xf32>
// CHECK:         }

// -----

func.func @test_reducesum2(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
  return %0 : tensor<3x1x2xf32>
}

// CHECK-LABEL:  func.func @test_reducesum2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>, [[PARAM_1_:%.+]]: tensor<?xi64>) -> tensor<3x1x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [3, 1, 2] : tensor<3xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies stablehlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.dynamic_reshape [[VAR_2_]], [[VAR_0_]] : (tensor<3x2xf32>, tensor<3xindex>) -> tensor<3x1x2xf32>
// CHECK:           return [[VAR_3_]] : tensor<3x1x2xf32>
// CHECK:         }

// -----

func.func @test_reducemean_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_reducemean_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>) -> tensor<3x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<2.000000e+00> : tensor<3x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies stablehlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.divide [[VAR_2_]], [[VAR_0_]] : tensor<3x2xf32>
// CHECK:           return [[VAR_3_]] : tensor<3x2xf32>
// CHECK:         }

// -----

func.func @test_reducemean_v13_2(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<?x?xf32>
  "func.return"(%0) : (tensor<?x?xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_reducemean_v13_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies stablehlo.add across dimensions = [1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x?x?xf32> -> tensor<3xindex>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_0_]], [[VAR_3_]], dims = [] : (tensor<f32>, tensor<3xindex>) -> tensor<?x?x?xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.reduce([[VAR_4_]] init: [[VAR_1_]]) applies stablehlo.add across dimensions = [1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           [[VAR_6_:%.+]] = stablehlo.divide [[VAR_2_]], [[VAR_5_]] : tensor<?x?xf32>
// CHECK:           return [[VAR_6_]] : tensor<?x?xf32>
// CHECK:         }

// -----

func.func @reduce_mean(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
  %0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.ReduceMean"(%arg0, %0) : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
  return %1 : tensor<2x5x1x1xf32>
}

// CHECK-LABEL:  func.func @reduce_mean
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 5, 1, 1] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<9.900000e+01> : tensor<2x5x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_2_]]) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x5x9x11xf32>, tensor<f32>) -> tensor<2x5xf32>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.dynamic_reshape [[VAR_3_]], [[VAR_0_]] : (tensor<2x5xf32>, tensor<4xindex>) -> tensor<2x5x1x1xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.divide [[VAR_4_]], [[VAR_1_]] : tensor<2x5x1x1xf32>
// CHECK:           return [[VAR_5_]] : tensor<2x5x1x1xf32>
// CHECK:         }

// -----

func.func @reduce_max(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
  %0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.ReduceMax"(%arg0, %0) : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
  return %1 : tensor<2x5x1x1xf32>
}

// CHECK-LABEL:  func.func @reduce_max
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 5, 1, 1] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies stablehlo.maximum across dimensions = [2, 3] : (tensor<2x5x9x11xf32>, tensor<f32>) -> tensor<2x5xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.dynamic_reshape [[VAR_2_]], [[VAR_0_]] : (tensor<2x5xf32>, tensor<4xindex>) -> tensor<2x5x1x1xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x5x1x1xf32>
// CHECK:         }

// -----


func.func @reduce_min(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
  %0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.ReduceMin"(%arg0, %0) : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
  return %1 : tensor<2x5x1x1xf32>
}

// CHECK-LABEL:  func.func @reduce_min
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 5, 1, 1] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies stablehlo.minimum across dimensions = [2, 3] : (tensor<2x5x9x11xf32>, tensor<f32>) -> tensor<2x5xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.dynamic_reshape [[VAR_2_]], [[VAR_0_]] : (tensor<2x5xf32>, tensor<4xindex>) -> tensor<2x5x1x1xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x5x1x1xf32>
// CHECK:         }
