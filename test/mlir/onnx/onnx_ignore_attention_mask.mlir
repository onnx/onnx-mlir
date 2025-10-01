// RUN: onnx-mlir-opt --ignore-attention-mask="arg-idx=1" %s -split-input-file | FileCheck %s

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
