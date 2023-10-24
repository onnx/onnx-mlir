// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_squeeze
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x1x32x1x64xf32>) -> tensor<16x32x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [16, 32, 64] : tensor<3xindex>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<16x1x32x1x64xf32>, tensor<3xindex>) -> tensor<16x32x64xf32>
// CHECK:           return [[VAR_1_]] : tensor<16x32x64xf32>
// CHECK:         }

// -----

func.func @test_squeeze_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<?x1x32x?x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_squeeze_unknown_dimensions
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x1x32x?x64xf32>) -> tensor<?x32x64xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x1x32x?x64xf32> -> tensor<5xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_0_]] : tensor<5xindex>, index -> index
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_2_]] : tensor<5xindex>, index -> index
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_4_]] : tensor<5xindex>, index -> index
// CHECK:           [[VAR_4_:%.+]] = shape.from_extents [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] : index, index, index
// CHECK:           [[VAR_5_:%.+]] = shape.to_extent_tensor [[VAR_4_]] : !shape.shape -> tensor<3xindex>
// CHECK:           [[VAR_6_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_5_]] : (tensor<?x1x32x?x64xf32>, tensor<3xindex>) -> tensor<?x32x64xf32>
// CHECK:           return [[VAR_6_]] : tensor<?x32x64xf32>
// CHECK:         }
