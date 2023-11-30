// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo --canonicalize --cse %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test the reshape op inference when constants are present.
//===----------------------------------------------------------------------===//

// -----

func.func @test_reshape_dynamic(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_reshape_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>, [[PARAM_1_:%.+]]: tensor<4xi64>) -> tensor<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_800_:%.+]] = arith.constant 800 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_1_]] : tensor<4xi64> to tensor<4xindex>
// CHECK:           [[VAR_1_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_0_]] : tensor<4xindex>, index -> index
// CHECK:           [[VAR_2_:%.+]] = arith.cmpi eq, [[VAR_1_]], [[CST_0_]] : index
// CHECK:           [[VAR_3_:%.+]] = arith.select [[VAR_2_]], [[CST_5_]], [[VAR_1_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.cmpi eq, [[VAR_3_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[CST_1_]], [[VAR_3_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_1_]] : tensor<4xindex>, index -> index
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_6_]], [[CST_0_]] : index
// CHECK:           [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_5_]], [[VAR_6_]] : index
// CHECK:           [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_8_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_1_]], [[VAR_8_]] : index
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.muli [[VAR_5_]], [[VAR_10_]] : index
// CHECK-DAG:       [[VAR_12_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_2_]] : tensor<4xindex>, index -> index
// CHECK:           [[VAR_13_:%.+]] = arith.cmpi eq, [[VAR_12_]], [[CST_0_]] : index
// CHECK:           [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[CST_1_]], [[VAR_12_]] : index
// CHECK:           [[VAR_15_:%.+]] = arith.cmpi eq, [[VAR_14_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_16_:%.+]] = arith.ori [[VAR_15_]], [[VAR_13_]] : i1
// CHECK:           [[VAR_17_:%.+]] = arith.select [[VAR_16_]], [[CST_1_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.muli [[VAR_11_]], [[VAR_17_]] : index
// CHECK-DAG:       [[VAR_19_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_3_]] : tensor<4xindex>, index -> index
// CHECK:           [[VAR_20_:%.+]] = arith.cmpi eq, [[VAR_19_]], [[CST_0_]] : index
// CHECK:           [[VAR_21_:%.+]] = arith.select [[VAR_20_]], [[CST_32_]], [[VAR_19_]] : index
// CHECK:           [[VAR_22_:%.+]] = arith.cmpi eq, [[VAR_21_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[CST_1_]], [[VAR_21_]] : index
// CHECK:           [[VAR_24_:%.+]] = arith.muli [[VAR_18_]], [[VAR_23_]] : index
// CHECK:           [[VAR_25_:%.+]] = arith.floordivsi [[CST_800_]], [[VAR_24_]] : index
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.select [[VAR_4_]], [[VAR_25_]], [[VAR_3_]] : index
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.select [[VAR_9_]], [[VAR_25_]], [[VAR_8_]] : index
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.select [[VAR_15_]], [[VAR_25_]], [[VAR_14_]] : index
// CHECK-DAG:       [[VAR_29_:%.+]] = arith.select [[VAR_22_]], [[VAR_25_]], [[VAR_21_]] : index
// CHECK:           [[VAR_30_:%.+]] = shape.from_extents [[VAR_26_]], [[VAR_27_]], [[VAR_28_]], [[VAR_29_]] : index, index, index, index
// CHECK:           [[VAR_31_:%.+]] = shape.to_extent_tensor [[VAR_30_]] : !shape.shape -> tensor<4xindex>
// CHECK:           [[VAR_32_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_31_]] : (tensor<5x5x1x32xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
// CHECK:           return [[VAR_32_]] : tensor<?x?x?x?xf32>

// -----

func.func @test_reshape_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[5, 5, 16, 2]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_reshape_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<5x5x16x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [5, 5, 16, 2] : tensor<4xindex>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x1x32xf32>, tensor<4xindex>) -> tensor<5x5x16x2xf32>
// CHECK:           return [[VAR_1_]] : tensor<5x5x16x2xf32>
// CHECK:         }

// -----

func.func @test_reshape_2(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 16, 2]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_reshape_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<25x16x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [25, 16, 2] : tensor<3xindex>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x1x32xf32>, tensor<3xindex>) -> tensor<25x16x2xf32>
// CHECK:           return [[VAR_1_]] : tensor<25x16x2xf32>
// CHECK:         }

// -----

func.func @test_reshape_3(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 0, 2]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_reshape_3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<80x5x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [80, 5, 2] : tensor<3xindex>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x1x32xf32>, tensor<3xindex>) -> tensor<80x5x2xf32>
// CHECK:           return [[VAR_1_]] : tensor<80x5x2xf32>
// CHECK:         }
