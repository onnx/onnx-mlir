// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s


func.func @test_where(%arg0: tensor<13x21x1xi1>, %arg1: tensor<13x21x1xf32>, %arg2: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Where"(%arg0, %arg1, %arg2) : (tensor<13x21x1xi1>, tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_where
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi1>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>, [[PARAM_2_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.select [[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]] : (tensor<13x21x1xi1>, tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_where_broadcast(%arg0: tensor<21x1xi1>, %arg1: tensor<13x21x1xf32>, %arg2: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Where"(%arg0, %arg1, %arg2) : (tensor<21x1xi1>, tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func.func @test_where_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<21x1xi1>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>, [[PARAM_2_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 21, 1>} : (tensor<21x1xi1>) -> tensor<1x21x1xi1>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.select [[VAR_0_]], [[PARAM_1_]], [[VAR_1_]] : (tensor<1x21x1xi1>, tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
// CHECK:           return [[VAR_2_]] : tensor<13x21x1xf32>
}

// -----

func.func @test_where_ui32(%arg0: tensor<13x21x1xi1>, %arg1: tensor<13x21x1xui32>, %arg2: tensor<13x21x1xui32>) -> tensor<13x21x1xui32> {
  %0 = "onnx.Where"(%arg0, %arg1, %arg2) : (tensor<13x21x1xi1>, tensor<13x21x1xui32>, tensor<13x21x1xui32>) -> tensor<13x21x1xui32>
  "func.return"(%0) : (tensor<13x21x1xui32>) -> ()
// CHECK-LABEL:  func.func @test_where_ui32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi1>, [[PARAM_1_:%.+]]: tensor<13x21x1xui32>, [[PARAM_2_:%.+]]: tensor<13x21x1xui32>) -> tensor<13x21x1xui32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.select [[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]] : (tensor<13x21x1xi1>, tensor<13x21x1xui32>, tensor<13x21x1xui32>) -> tensor<13x21x1xui32>
// CHECK:           return [[VAR_0_]] : tensor<13x21x1xui32>
}

// -----

func.func @test_where_f64(%arg0: tensor<13x21x1xi1>, %arg1: tensor<13x21x1xf64>, %arg2: tensor<13x21x1xf64>) -> tensor<13x21x1xf64> {
  %0 = "onnx.Where"(%arg0, %arg1, %arg2) : (tensor<13x21x1xi1>, tensor<13x21x1xf64>, tensor<13x21x1xf64>) -> tensor<13x21x1xf64>
  "func.return"(%0) : (tensor<13x21x1xf64>) -> ()
// CHECK-LABEL:  func.func @test_where_f64
// CHECK-NOT:    onnx.Where
// CHECK:        return {{.*}}: tensor<13x21x1xf64>
}