// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_depth_to_space(%arg0 : tensor<2x16x20x20xf32>) -> tensor<2x4x40x40xf32> {
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "CRD"} : (tensor<2x16x20x20xf32>) -> tensor<2x4x40x40xf32>
  "func.return"(%0) : (tensor<2x4x40x40xf32>) -> ()
// CHECK-LABEL:  func.func @test_depth_to_space
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x16x20x20xf32>) -> tensor<2x4x40x40xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 4, 40, 40] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.const_shape [2, 4, 2, 2, 20, 20] : tensor<6xindex>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_1_]] : (tensor<2x16x20x20xf32>, tensor<6xindex>) -> tensor<2x4x2x2x20x20xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.transpose [[VAR_2_]], dims = [0, 1, 4, 2, 5, 3] : (tensor<2x4x2x2x20x20xf32>) -> tensor<2x4x20x2x20x2xf32>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.dynamic_reshape [[VAR_3_]], [[VAR_0_]] : (tensor<2x4x20x2x20x2xf32>, tensor<4xindex>) -> tensor<2x4x40x40xf32>
// CHECK:           return [[VAR_4_]] : tensor<2x4x40x40xf32>
// CHECK:         }
}

// -----

func.func @test_depth_to_space_dynamic(%arg0 : tensor<2x?x20x?xf32>) -> tensor<2x?x40x?xf32> {
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "CRD"} : (tensor<2x?x20x?xf32>) -> tensor<2x?x40x?xf32>
  "func.return"(%0) : (tensor<2x?x40x?xf32>) -> ()
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-LABEL:  func.func @test_depth_to_space_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x?x20x?xf32>) -> tensor<2x?x40x?xf32> {
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<2x?x20x?xf32> -> tensor<4xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_1_]] : tensor<4xindex>, index -> index
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<2x?x20x?xf32> -> tensor<4xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.get_extent [[VAR_2_]], [[CST_3_]] : tensor<4xindex>, index -> index
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_3_]]{{.}}
// CHECK-DAG:       [[VAR_6_:%.+]] = shape.from_extents [[CST_2_]], [[VAR_4_]], [[CST_2_]], [[CST_2_]], [[CST_2_]]0, [[VAR_3_]] : index, index, index, index, index, index
// CHECK:           [[VAR_7_:%.+]] = shape.to_extent_tensor [[VAR_6_]] : !shape.shape -> tensor<6xindex>
// CHECK:           [[VAR_8_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_7_]] : (tensor<2x?x20x?xf32>, tensor<6xindex>) -> tensor<2x?x2x2x20x?xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.transpose [[VAR_8_]], dims = [0, 1, 4, 2, 5, 3] : (tensor<2x?x2x2x20x?xf32>) -> tensor<2x?x20x2x?x2xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = shape.from_extents [[CST_2_]], [[VAR_4_]], [[CST_40_]], [[VAR_5_]] : index, index, index, index
// CHECK:           [[VAR_11_:%.+]] = shape.to_extent_tensor [[VAR_10_]] : !shape.shape -> tensor<4xindex>
// CHECK:           [[VAR_12_:%.+]] = stablehlo.dynamic_reshape [[VAR_9_]], [[VAR_11_]] : (tensor<2x?x20x2x?x2xf32>, tensor<4xindex>) -> tensor<2x?x40x?xf32>
// CHECK:           return [[VAR_12_]] : tensor<2x?x40x?xf32>
// CHECK:         }
}
