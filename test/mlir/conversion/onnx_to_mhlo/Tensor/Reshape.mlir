// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --canonicalize --cse %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test the reshape op inference when constants are present.
//===----------------------------------------------------------------------===//

func.func @test_reshape_dynamic(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: func.func @test_reshape_dynamic
  // CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>, [[PARAM_1_:%.+]]: tensor<4xi64>) -> tensor<?x?x?x?xf32> {
  // CHECK-DAG: %c3 = arith.constant 3 : index
  // CHECK-DAG: %c2 = arith.constant 2 : index
  // CHECK-DAG: %c-1 = arith.constant -1 : index
  // CHECK-DAG: %c0 = arith.constant 0 : index
  // CHECK-DAG: %c1 = arith.constant 1 : index
  // CHECK-DAG: %c5 = arith.constant 5 : index
  // CHECK-DAG: %c32 = arith.constant 32 : index
  // CHECK-DAG: %c800 = arith.constant 800 : index
  // CHECK: [[VAR_0_:%.+]] = arith.index_cast [[PARAM_1_]] : tensor<4xi64> to tensor<4xindex>
  // CHECK: [[VAR_1_:%.+]] = shape.get_extent [[VAR_0_]], %c0 : tensor<4xindex>, index -> index
  // CHECK: [[VAR_2_:%.+]] = arith.cmpi eq, [[VAR_1_]], %c0 : index
  // CHECK: [[VAR_3_:%.+]] = arith.select [[VAR_2_]], %c5, [[VAR_1_]] : index
  // CHECK: [[VAR_4_:%.+]] = arith.cmpi eq, [[VAR_3_]], %c-1 : index
  // CHECK: [[VAR_5_:%.+]] = arith.select [[VAR_4_]], %c1, [[VAR_3_]] : index
  // CHECK: [[VAR_6_:%.+]] = shape.get_extent [[VAR_0_]], %c1 : tensor<4xindex>, index -> index
  // CHECK: [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_6_]], %c0 : index
  // CHECK: [[VAR_8_:%.+]] = arith.select [[VAR_7_]], %c5, [[VAR_6_]] : index
  // CHECK: [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_8_]], %c-1 : index
  // CHECK: [[VAR_10_:%.+]] = arith.select [[VAR_9_]], %c1, [[VAR_8_]] : index
  // CHECK: [[VAR_11_:%.+]] = arith.muli [[VAR_5_]], [[VAR_10_]] : index
  // CHECK: [[VAR_12_:%.+]] = shape.get_extent [[VAR_0_]], %c2 : tensor<4xindex>, index -> index
  // CHECK: [[VAR_13_:%.+]] = arith.cmpi eq, [[VAR_12_]], %c0 : index
  // CHECK: [[VAR_14_:%.+]] = arith.select [[VAR_13_]], %c1, [[VAR_12_]] : index
  // CHECK: [[VAR_15_:%.+]] = arith.cmpi eq, [[VAR_14_]], %c-1 : index
  // CHECK: [[VAR_16_:%.+]] = arith.select [[VAR_15_]], %c1, [[VAR_14_]] : index
  // CHECK: [[VAR_17_:%.+]] = arith.muli [[VAR_11_]], [[VAR_16_]] : index
  // CHECK: [[VAR_18_:%.+]] = shape.get_extent [[VAR_0_]], %c3 : tensor<4xindex>, index -> index
  // CHECK: [[VAR_19_:%.+]] = arith.cmpi eq, [[VAR_18_]], %c0 : index
  // CHECK: [[VAR_20_:%.+]] = arith.select [[VAR_19_]], %c32, [[VAR_18_]] : index
  // CHECK: [[VAR_21_:%.+]] = arith.cmpi eq, [[VAR_20_]], %c-1 : index
  // CHECK: [[VAR_22_:%.+]] = arith.select [[VAR_21_]], %c1, [[VAR_20_]] : index
  // CHECK: [[VAR_23_:%.+]] = arith.muli [[VAR_17_]], [[VAR_22_]] : index
  // CHECK: [[VAR_24_:%.+]] = arith.floordivsi %c800, [[VAR_23_]] : index
  // CHECK: [[VAR_25_:%.+]] = arith.select [[VAR_4_]], [[VAR_24_]], [[VAR_3_]] : index
  // CHECK: [[VAR_26_:%.+]] = arith.select [[VAR_9_]], [[VAR_24_]], [[VAR_8_]] : index
  // CHECK: [[VAR_27_:%.+]] = arith.select [[VAR_15_]], [[VAR_24_]], [[VAR_14_]] : index
  // CHECK: [[VAR_28_:%.+]] = arith.select [[VAR_21_]], [[VAR_24_]], [[VAR_20_]] : index
  // CHECK: [[VAR_29_:%.+]] = shape.from_extents [[VAR_25_]], [[VAR_26_]], [[VAR_27_]], [[VAR_28_]] : index, index, index, index
  // CHECK: [[VAR_30_:%.+]] = shape.to_extent_tensor [[VAR_29_]] : !shape.shape -> tensor<4xindex>
  // CHECK: [[VAR_31_:%.+]] = mhlo.dynamic_reshape [[PARAM_0_]], [[VAR_30_]] : (tensor<5x5x1x32xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
}

// -----

func.func @test_reshape_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[5, 5, 16, 2]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reshape_1
  // CHECK: %0 = mhlo.reshape %arg0 : (tensor<5x5x1x32xf32>) -> tensor<5x5x16x2xf32>
}

// -----

func.func @test_reshape_2(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 16, 2]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reshape_2
  // CHECK: %0 = mhlo.reshape %arg0 : (tensor<5x5x1x32xf32>) -> tensor<25x16x2xf32>
}

// -----

func.func @test_reshape_3(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 0, 2]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reshape_3
  // CHECK: %0 = mhlo.reshape %arg0 : (tensor<5x5x1x32xf32>) -> tensor<80x5x2xf32>
}

// -----
