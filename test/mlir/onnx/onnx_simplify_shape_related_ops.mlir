// RUN: onnx-mlir-opt --simplify-shape-related-ops-onnx %s -split-input-file | FileCheck %s

func.func @test_shape_to_dim(%arg0: tensor<?x256xi64>) -> (tensor<2xi64>) {
  %0 = "onnx.Shape"(%arg0) : (tensor<?x256xi64>) -> tensor<2xi64>
  return %0 : tensor<2xi64>

// CHECK-LABEL:  func.func @test_shape_to_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x256xi64>) -> tensor<2xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           return [[VAR_2_]] : tensor<2xi64>
// CHECK:         }
}

// -----

func.func @test_shape_to_dim_positive_axis(%arg0: tensor<?x256x?xi64>) -> (tensor<2xi64>) {
  %0 = "onnx.Shape"(%arg0) {start = 0 : si64, end = 2 : si64} : (tensor<?x256x?xi64>) -> tensor<2xi64>
  return %0 : tensor<2xi64>

// CHECK-LABEL:  func.func @test_shape_to_dim_positive_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x256x?xi64>) -> tensor<2xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x256x?xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           return [[VAR_2_]] : tensor<2xi64>
// CHECK:         }
}

// -----

func.func @test_shape_to_dim_negative_axis(%arg0: tensor<?x256x?xi64>) -> (tensor<2xi64>) {
  %0 = "onnx.Shape"(%arg0) {start = -2 : si64, end = 3 : si64} : (tensor<?x256x?xi64>) -> tensor<2xi64>
  return %0 : tensor<2xi64>

// CHECK-LABEL:  func.func @test_shape_to_dim_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x256x?xi64>) -> tensor<2xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 2 : si64} : (tensor<?x256x?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           return [[VAR_2_]] : tensor<2xi64>
// CHECK:         }
}

// -----

func.func @test_pass_dims_through_cast(%arg0: tensor<?x256xi64>) -> (tensor<2xf32>) {
  %0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  %1 = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
  %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %3 = "onnx.Cast"(%2) {to = f32} : (tensor<2xi64>) -> tensor<2xf32>
  return %3 : tensor<2xf32>

// CHECK-LABEL:  func.func @test_pass_dims_through_cast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x256xi64>) -> tensor<2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Cast"([[VAR_0_]]) {to = f32} : (tensor<1xi64>) -> tensor<1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Constant"() {value = dense<2.560000e+02> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_2_]]) {axis = 0 : si64} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// CHECK:           return [[VAR_3_]] : tensor<2xf32>
// CHECK:         }
}

// -----

func.func @test_pass_dims_through_concat(%arg0: tensor<?x256xi64>) -> (tensor<4xi64>) {
  %0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  %1 = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
  %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %3 = "onnx.Concat"(%2, %0) {axis = 0 : si64} : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
  %4 = "onnx.Concat"(%3, %0) {axis = 0 : si64} : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
  return %4 : tensor<4xi64>

// CHECK-LABEL:  func.func @test_pass_dims_through_concat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x256xi64>) -> tensor<4xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]], [[VAR_0_]], [[VAR_0_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           return [[VAR_2_]] : tensor<4xi64>
// CHECK:         }
}

// -----

func.func @test_pass_dims_through_cast(%arg0: tensor<?x?x200xf32>) -> tensor<2xi64> {
  %0 = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x?x200xf32>) -> tensor<1xi64>
  %2 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<?x?x200xf32>) -> tensor<1xi64>
  %3 = "onnx.Constant"() {value = dense<200> : tensor<1xi64>} : () -> tensor<1xi64>
  %4 = "onnx.Concat"(%1, %2, %3) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %5 = "onnx.Gather"(%4, %0) {axis = 0 : si64} : (tensor<3xi64>, tensor<2xi64>) -> tensor<2xi64>
  return %5 : tensor<2xi64>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_pass_dims_through_cast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x200xf32>) -> tensor<2xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x?x200xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64} : (tensor<?x?x200xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           return [[VAR_2_]] : tensor<2xi64>
// CHECK:         }
}

// -----

func.func @test_pass_dims_through_slice(%arg0: tensor<?x256xi64>) -> (tensor<1xi64>) {
  %0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  %1 = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
  %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %3 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %4 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %5 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %6 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %7 = "onnx.Slice"(%2, %3, %4, %5, %6) : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  return %7 : tensor<1xi64>

// CHECK-LABEL:  func.func @test_pass_dims_through_slice
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x256xi64>) -> tensor<1xi64> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK:           return [[VAR_0_]] : tensor<1xi64>
// CHECK:         }
}

// -----

func.func @test_update_reshape_output_shape(%arg0: tensor<?x256xi64>, %arg1: tensor<?x256xi64>) -> (tensor<?x?x?xi64>) {
  %0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %2 = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
  %3 = "onnx.Concat"(%0, %1, %2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %4 = "onnx.Reshape"(%arg1, %3) : (tensor<?x256xi64>, tensor<3xi64>) -> tensor<?x?x?xi64>
  return %4 : tensor<?x?x?xi64>

// CHECK-LABEL:  func.func @test_update_reshape_output_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x256xi64>, [[PARAM_1_:%.+]]: tensor<?x256xi64>) -> tensor<?x1x256xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_3_]]) : (tensor<?x256xi64>, tensor<3xi64>) -> tensor<?x1x256xi64>
// CHECK:           return [[VAR_4_]] : tensor<?x1x256xi64>
// CHECK:         }
}

// -----

func.func @test_update_constantofshape_output_shape(%arg0: tensor<?x256xi64>, %arg1: tensor<?x256xi64>) -> (tensor<?x?x?xi64>) {
  %0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %2 = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
  %3 = "onnx.Concat"(%0, %1, %2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %4 = "onnx.ConstantOfShape"(%3) {value = dense<1> : tensor<1xi64>} : (tensor<3xi64>) -> tensor<?x?x?xi64>
  return %4 : tensor<?x?x?xi64>

// CHECK-LABEL:  func.func @test_update_constantofshape_output_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x256xi64>, [[PARAM_1_:%.+]]: tensor<?x256xi64>) -> tensor<?x1x256xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ConstantOfShape"([[VAR_3_]]) {value = dense<1> : tensor<1xi64>} : (tensor<3xi64>) -> tensor<?x1x256xi64>
// CHECK:           return [[VAR_4_]] : tensor<?x1x256xi64>
// CHECK:         }
}
