// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --lower-affine --canonicalize %s -split-input-file | FileCheck %s

func.func @test_unsqueeze(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_unsqueeze    
// CHECK-NEXT:         %0 = mhlo.reshape %arg0 : (tensor<10x10xf32>) -> tensor<1x10x10x1xf32>
// CHECK-NEXT:         return %0 : tensor<1x10x10x1xf32>
}

func.func @test_unsqueeze_negative_axis(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<1xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_unsqueeze_negative_axis
// CHECK-NEXT:         %0 = mhlo.reshape %arg0 : (tensor<16x32x64xf32>) -> tensor<16x32x1x64xf32>
// CHECK-NEXT:         return %0 : tensor<16x32x1x64xf32>
}

func.func @test_unsqueeze_mix(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_unsqueeze_mix
// CHECK-NEXT:         %0 = mhlo.reshape %arg0 : (tensor<16x32x64xf32>) -> tensor<16x1x32x1x64xf32>
// CHECK-NEXT:         return %0 : tensor<16x1x32x1x64xf32>
}

func.func @test_unsqueeze_unknown_dimensions(%arg0 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<?x?xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_unsqueeze_unknown_dimensions
// CHECK-DAG:      [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:      [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:      [[VAR_0_:%.+]] = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
// CHECK-DAG:      [[VAR_1_:%.+]] = shape.get_extent [[VAR_0_]], [[C0]] : tensor<2xindex>, index -> index
// CHECK-DAG:      [[VAR_2_:%.+]] = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
// CHECK-DAG:      [[VAR_3_:%.+]] = shape.get_extent [[VAR_2_]], [[C1]] : tensor<2xindex>, index -> index
// CHECK-DAG:      [[VAR_4_:%.+]] = shape.from_extents [[C1]], [[VAR_1_]], [[C1]], [[VAR_3_]] : index, index, index, index
// CHECK-DAG:      [[VAR_5_:%.+]] = shape.to_extent_tensor [[VAR_4_]] : !shape.shape -> tensor<4xindex>
// CHECK-DAG:      [[VAR_6_:%.+]] = mhlo.dynamic_reshape %arg0, [[VAR_5_]] : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<1x?x1x?xf32>
// CHECK-DAG:      return [[VAR_6_]] : tensor<1x?x1x?xf32>
}
