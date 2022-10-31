// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --lower-affine --canonicalize %s -split-input-file | FileCheck %s

func.func @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_squeeze
// CHECK: %0 = mhlo.reshape %arg0 : (tensor<16x1x32x1x64xf32>) -> tensor<16x32x64xf32>
}

func.func @test_squeeze_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<?x1x32x?x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_squeeze_unknown_dimensions
// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[VAR_0_:%.+]] = shape.shape_of %arg0 : tensor<?x1x32x?x64xf32> -> tensor<5xindex>
// CHECK-DAG:    [[VAR_1_:%.+]] = shape.get_extent %0, [[C0]] : tensor<5xindex>, index -> index
// CHECK-DAG:    [[VAR_2_:%.+]] = shape.from_extents [[VAR_1_]], [[C32]], [[C64]] : index, index, index
// CHECK-DAG:    [[VAR_3_:%.+]] = shape.to_extent_tensor [[VAR_2_]] : !shape.shape -> tensor<3xindex>
// CHECK-DAG:    [[VAR_4_:%.+]] = mhlo.dynamic_reshape %arg0, [[VAR_3_]] : (tensor<?x1x32x?x64xf32>, tensor<3xindex>) -> tensor<?x32x64xf32>
}