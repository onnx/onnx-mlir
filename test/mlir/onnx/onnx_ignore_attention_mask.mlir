// RUN: onnx-mlir-opt --ignore-attention-mask="arg-idx=1" %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --ignore-attention-mask="arg-idx=2" %s -split-input-file | FileCheck %s --check-prefix=NOT-IGNORE

// COM: test ignoring attention_mask input (arg1) in the function, so that attention_mask is not used in the AddOp sandwitched beteen MatMulOp and SoftmaxOp.
func.func @test_ignore_attention_layer(%arg0: tensor<1x?xi64>, %arg1: tensor<1x?xi64>, %arg2: tensor<1x12x?x64xf32>, %arg3: tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
{
  %0 = onnx.Constant dense<1> : tensor<1xi64>
  %1 = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %2 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %3 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<1x?xi64>) -> tensor<1xi64>
  %4 = "onnx.Dim"(%arg1) {axis = 1 : si64} : (tensor<1x?xi64>) -> tensor<1xi64>
  %5 = "onnx.Unsqueeze"(%arg1, %0) : (tensor<1x?xi64>, tensor<1xi64>) -> tensor<1x1x?xi64>
  %6 = "onnx.Unsqueeze"(%5, %0) : (tensor<1x1x?xi64>, tensor<1xi64>) -> tensor<1x1x1x?xi64>
  %7 = "onnx.Concat"(%0, %0, %3, %4) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %8 = "onnx.Expand"(%6, %7) : (tensor<1x1x1x?xi64>, tensor<4xi64>) -> tensor<1x1x?x?xi64>
  %9 = "onnx.Cast"(%8) {saturate = 1 : si64, to = f32} : (tensor<1x1x?x?xi64>) -> tensor<1x1x?x?xf32>
  %10 = "onnx.Sub"(%2, %9) : (tensor<f32>, tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
  %11 = "onnx.Cast"(%10) {saturate = 1 : si64, to = i1} : (tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xi1>
  %12 = "onnx.Where"(%11, %1, %10) : (tensor<1x1x?x?xi1>, tensor<f32>, tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
  %13 = "onnx.MatMul"(%arg2, %arg3) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %14 = "onnx.Add"(%13, %12) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %15 = "onnx.Softmax"(%14) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  onnx.Return %15 : tensor<1x12x?x?xf32>

// CHECK-LABEL:  func.func @test_ignore_attention_layer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?xi64>, [[PARAM_1_:%.+]]: tensor<1x?xi64>, [[PARAM_2_:%.+]]: tensor<1x12x?x64xf32>, [[PARAM_3_:%.+]]: tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.MatMul"([[PARAM_2_]], [[PARAM_3_]]) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Softmax"([[VAR_0_]]) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<1x12x?x?xf32>
// CHECK:         }
}

// -----

// COM: test NOT ignoring attention_mask input (arg1) in the function, so that attention_mask is not used in the AddOp sandwitched beteen MatMulOp and SoftmaxOp.
func.func @test_ignore_attention_layer(%arg0: tensor<1x?xi64>, %arg1: tensor<1x?xi64>, %arg2: tensor<1x12x?x64xf32>, %arg3: tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
{
  %0 = onnx.Constant dense<1> : tensor<1xi64>
  %1 = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %2 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %3 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<1x?xi64>) -> tensor<1xi64>
  %4 = "onnx.Dim"(%arg1) {axis = 1 : si64} : (tensor<1x?xi64>) -> tensor<1xi64>
  %5 = "onnx.Unsqueeze"(%arg1, %0) : (tensor<1x?xi64>, tensor<1xi64>) -> tensor<1x1x?xi64>
  %6 = "onnx.Unsqueeze"(%5, %0) : (tensor<1x1x?xi64>, tensor<1xi64>) -> tensor<1x1x1x?xi64>
  %7 = "onnx.Concat"(%0, %0, %3, %4) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %8 = "onnx.Expand"(%6, %7) : (tensor<1x1x1x?xi64>, tensor<4xi64>) -> tensor<1x1x?x?xi64>
  %9 = "onnx.Cast"(%8) {saturate = 1 : si64, to = f32} : (tensor<1x1x?x?xi64>) -> tensor<1x1x?x?xf32>
  %10 = "onnx.Sub"(%2, %9) : (tensor<f32>, tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
  %11 = "onnx.Cast"(%10) {saturate = 1 : si64, to = i1} : (tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xi1>
  %12 = "onnx.Where"(%11, %1, %10) : (tensor<1x1x?x?xi1>, tensor<f32>, tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
  %13 = "onnx.MatMul"(%arg2, %arg3) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %14 = "onnx.Add"(%13, %12) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %15 = "onnx.Softmax"(%14) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  onnx.Return %15 : tensor<1x12x?x?xf32>

// NOT-IGNORE-LABEL:  func.func @test_ignore_attention_layer
// NOT-IGNORE-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?xi64>, [[PARAM_1_:%.+]]: tensor<1x?xi64>, [[PARAM_2_:%.+]]: tensor<1x12x?x64xf32>, [[PARAM_3_:%.+]]: tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32> {
// NOT-IGNORE-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// NOT-IGNORE-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// NOT-IGNORE-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// NOT-IGNORE-DAG:       [[VAR_3_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64} : (tensor<1x?xi64>) -> tensor<1xi64>
// NOT-IGNORE-DAG:       [[VAR_4_:%.+]] = "onnx.Dim"([[PARAM_1_]]) {axis = 1 : si64} : (tensor<1x?xi64>) -> tensor<1xi64>
// NOT-IGNORE:           [[VAR_5_:%.+]] = "onnx.Unsqueeze"([[PARAM_1_]], [[VAR_0_]]) : (tensor<1x?xi64>, tensor<1xi64>) -> tensor<1x1x?xi64>
// NOT-IGNORE-DAG:       [[VAR_6_:%.+]] = "onnx.Unsqueeze"([[VAR_5_]], [[VAR_0_]]) : (tensor<1x1x?xi64>, tensor<1xi64>) -> tensor<1x1x1x?xi64>
// NOT-IGNORE-DAG:       [[VAR_7_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_0_]], [[VAR_3_]], [[VAR_4_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// NOT-IGNORE:           [[VAR_8_:%.+]] = "onnx.Expand"([[VAR_6_]], [[VAR_7_]]) : (tensor<1x1x1x?xi64>, tensor<4xi64>) -> tensor<1x1x?x?xi64>
// NOT-IGNORE:           [[VAR_9_:%.+]] = "onnx.Cast"([[VAR_8_]]) {saturate = 1 : si64, to = f32} : (tensor<1x1x?x?xi64>) -> tensor<1x1x?x?xf32>
// NOT-IGNORE:           [[VAR_10_:%.+]] = "onnx.Sub"([[VAR_2_]], [[VAR_9_]]) : (tensor<f32>, tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
// NOT-IGNORE:           [[VAR_11_:%.+]] = "onnx.Cast"([[VAR_10_]]) {saturate = 1 : si64, to = i1} : (tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xi1>
// NOT-IGNORE-DAG:       [[VAR_12_:%.+]] = "onnx.Where"([[VAR_11_]], [[VAR_1_]], [[VAR_1_]]0) : (tensor<1x1x?x?xi1>, tensor<f32>, tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
// NOT-IGNORE-DAG:       [[VAR_13_:%.+]] = "onnx.MatMul"([[PARAM_2_]], [[PARAM_3_]]) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
// NOT-IGNORE:           [[VAR_14_:%.+]] = "onnx.Add"([[VAR_13_]], [[VAR_12_]]) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
// NOT-IGNORE:           [[VAR_15_:%.+]] = "onnx.Softmax"([[VAR_14_]]) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
// NOT-IGNORE:           onnx.Return [[VAR_15_]] : tensor<1x12x?x?xf32>
// NOT-IGNORE:         }
}
