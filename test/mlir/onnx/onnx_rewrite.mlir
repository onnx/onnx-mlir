// RUN: onnx-mlir-opt --onnx-rewrite %s -split-input-file | FileCheck %s

// Check replacing binary ops by ConstantOfShape

func.func @test_replace_add_by_constantofshape_1(%arg0: tensor<?xi64>) -> tensor<2x?xf32> {
    %0 = onnx.Constant dense<2> : tensor<1xi64>
    %1 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
    %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %4 = onnx.ConstantOfShape(%2) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
    %5 = "onnx.Add"(%3, %4) : (tensor<f32>, tensor<2x?xf32>) -> tensor<2x?xf32>
    return %5: tensor<2x?xf32>

// CHECK-LABEL:  func.func @test_replace_add_by_constantofshape_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<2x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = onnx.ConstantOfShape([[VAR_2_]]) {value = dense<2.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x?xf32>
// CHECK:         }
}

// -----

func.func @test_replace_add_by_constantofshape_2(%arg0: tensor<?xi64>) -> tensor<2x?xf32> {
    %0 = onnx.Constant dense<2> : tensor<1xi64>
    %1 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
    %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = onnx.ConstantOfShape(%2) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
    %4 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %5 = "onnx.Add"(%3, %4) : (tensor<2x?xf32>, tensor<f32>) -> tensor<2x?xf32>
    return %5: tensor<2x?xf32>

// CHECK-LABEL:  func.func @test_replace_add_by_constantofshape_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<2x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = onnx.ConstantOfShape([[VAR_2_]]) {value = dense<2.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x?xf32>
// CHECK:         }
}

// -----

func.func @test_replace_div_by_constantofshape_1(%arg0: tensor<?xi64>) -> tensor<2x?xf32> {
    %0 = onnx.Constant dense<2> : tensor<1xi64>
    %1 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
    %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = onnx.Constant dense<5.000000e+00> : tensor<f32>
    %4 = onnx.ConstantOfShape(%2) {value = dense<2.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
    %5 = "onnx.Div"(%3, %4) : (tensor<f32>, tensor<2x?xf32>) -> tensor<2x?xf32>
    return %5: tensor<2x?xf32>

// CHECK-LABEL:  func.func @test_replace_div_by_constantofshape_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<2x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = onnx.ConstantOfShape([[VAR_2_]]) {value = dense<2.500000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x?xf32>
// CHECK:         }
}

// -----

func.func @test_replace_div_by_constantofshape_2(%arg0: tensor<?xi64>) -> tensor<2x?xf32> {
    %0 = onnx.Constant dense<2> : tensor<1xi64>
    %1 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
    %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = onnx.ConstantOfShape(%2) {value = dense<5.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
    %4 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %5 = "onnx.Div"(%3, %4) : (tensor<2x?xf32>, tensor<f32>) -> tensor<2x?xf32>
    return %5: tensor<2x?xf32>

// CHECK-LABEL:  func.func @test_replace_div_by_constantofshape_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<2x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = onnx.ConstantOfShape([[VAR_2_]]) {value = dense<2.500000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x?xf32>
// CHECK:         }
}

// -----

func.func @test_replace_mul_by_constantofshape_1(%arg0: tensor<?xi64>) -> tensor<2x?xf32> {
    %0 = onnx.Constant dense<2> : tensor<1xi64>
    %1 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
    %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = onnx.Constant dense<5.000000e+00> : tensor<f32>
    %4 = onnx.ConstantOfShape(%2) {value = dense<2.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
    %5 = "onnx.Mul"(%3, %4) : (tensor<f32>, tensor<2x?xf32>) -> tensor<2x?xf32>
    return %5: tensor<2x?xf32>

// CHECK-LABEL:  func.func @test_replace_mul_by_constantofshape_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<2x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = onnx.ConstantOfShape([[VAR_2_]]) {value = dense<1.000000e+01> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x?xf32>
// CHECK:         }
}

// -----

func.func @test_replace_mul_by_constantofshape_2(%arg0: tensor<?xi64>) -> tensor<2x?xf32> {
    %0 = onnx.Constant dense<2> : tensor<1xi64>
    %1 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
    %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = onnx.ConstantOfShape(%2) {value = dense<5.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
    %4 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %5 = "onnx.Mul"(%3, %4) : (tensor<2x?xf32>, tensor<f32>) -> tensor<2x?xf32>
    return %5: tensor<2x?xf32>

// CHECK-LABEL:  func.func @test_replace_mul_by_constantofshape_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<2x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = onnx.ConstantOfShape([[VAR_2_]]) {value = dense<1.000000e+01> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x?xf32>
// CHECK:         }
}

// -----

func.func @test_replace_sub_by_constantofshape_1(%arg0: tensor<?xi64>) -> tensor<2x?xf32> {
    %0 = onnx.Constant dense<2> : tensor<1xi64>
    %1 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
    %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = onnx.Constant dense<5.000000e+00> : tensor<f32>
    %4 = onnx.ConstantOfShape(%2) {value = dense<2.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
    %5 = "onnx.Sub"(%3, %4) : (tensor<f32>, tensor<2x?xf32>) -> tensor<2x?xf32>
    return %5: tensor<2x?xf32>

// CHECK-LABEL:  func.func @test_replace_sub_by_constantofshape_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<2x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = onnx.ConstantOfShape([[VAR_2_]]) {value = dense<3.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x?xf32>
// CHECK:         }
}

// -----

func.func @test_replace_sub_by_constantofshape_2(%arg0: tensor<?xi64>) -> tensor<2x?xf32> {
    %0 = onnx.Constant dense<2> : tensor<1xi64>
    %1 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
    %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = onnx.ConstantOfShape(%2) {value = dense<5.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
    %4 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %5 = "onnx.Sub"(%3, %4) : (tensor<2x?xf32>, tensor<f32>) -> tensor<2x?xf32>
    return %5: tensor<2x?xf32>

// CHECK-LABEL:  func.func @test_replace_sub_by_constantofshape_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<2x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = onnx.ConstantOfShape([[VAR_2_]]) {value = dense<3.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<2x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x?xf32>
// CHECK:         }
}

// -----

func.func @test_replace_unsqueeze_by_constantofshape(%arg0: tensor<1x?xi64>) -> tensor<1x1x1x?xf32> {
    %0 = onnx.Constant dense<1> : tensor<1xi64>
    %1 = onnx.Constant dense<2> : tensor<1xi64>
    %2 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<1x?xi64>) -> tensor<1xi64>
    %3 = "onnx.Concat"(%0, %2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %4 = onnx.ConstantOfShape(%3) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<1x?xf32>
    %5 = "onnx.Unsqueeze"(%4, %0) : (tensor<1x?xf32>, tensor<1xi64>) -> tensor<1x1x?xf32>
    %6 = "onnx.Unsqueeze"(%5, %1) : (tensor<1x1x?xf32>, tensor<1xi64>) -> tensor<1x1x1x?xf32>
    return %6 : tensor<1x1x1x?xf32>

// CHECK-LABEL:  func.func @test_replace_unsqueeze_by_constantofshape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?xi64>) -> tensor<1x1x1x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64} : (tensor<1x?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_0_]], [[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_3_:%.+]] = onnx.ConstantOfShape([[VAR_2_]]) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<4xi64>) -> tensor<1x1x1x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x1x1x?xf32>
// CHECK:         }
}

// -----

