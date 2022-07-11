// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s -split-input-file | FileCheck %s

func.func @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = mhlo.add [[PARAM_0_]], [[PARAM_1_]] : tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

func.func @test_add_dynamic(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_add_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>, [[PARAM_1_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = mhlo.add [[PARAM_0_]], [[PARAM_1_]] : tensor<?x10xf32>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<?x10xf32>
// CHECK-NEXT:   }
}

func.func @test_relu(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<10x10xf32>
// CHECK-NEXT:     [[VAR_1_:%.+]] = mhlo.maximum [[PARAM_0_]], [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:     return [[VAR_1_]] : tensor<10x10xf32>
// CHECK-NEXT:   }
}

func.func @test_relu_dynamic(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_relu_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-DAG:      [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      [[VAR_1_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK-NEXT:     [[VAR_2_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_0_]], [[VAR_1_]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK-NEXT:     [[VAR_3_:%.+]] = mhlo.maximum [[PARAM_0_]], [[VAR_2_]] : tensor<?x10xf32>
// CHECK-NEXT:     return [[VAR_3_]] : tensor<?x10xf32>
// CHECK-NEXT:   }
}

