// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

func @test_lstm_all_timesteps(%X: tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %W: tensor<1x8x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %R: tensor<1x16x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> () {
  %cst = "onnx.NoValue"() {value} : () -> none
  %hn_output, %cf_output = "zhigh.LSTM"(%X, %cst, %cst, %W, %cst, %R, %cst) {direction = "forward", hidden_size = 16 : si64, return_all_steps = -1 : si64} : (tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none, none, tensor<1x8x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, none, tensor<1x16x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, none) -> (tensor<*xf32>, tensor<*xf32>)
  return

// mlir2FileCheck.py -a'["X", "W", "R"]'
// CHECK-LABEL:  func @test_lstm_all_timesteps
// CHECK-SAME:   ([[X_:%.+]]: tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, [[W_:%.+]]: tensor<1x8x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, [[R_:%.+]]: tensor<1x16x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[X_]], [[VAR_cst_]], [[VAR_cst_]], [[W_]], [[VAR_cst_]], [[R_]], [[VAR_cst_]]) {direction = "forward", hidden_size = 16 : si64, return_all_steps = -1 : si64} : (tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none, none, tensor<1x8x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, none, tensor<1x16x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, none) -> (tensor<2x1x4x16xf32, #zhigh.encoding<{dataLayout = "4DS"}>>, tensor<1x1x4x16xf32, #zhigh.encoding<{dataLayout = "4DS"}>>)
// CHECK:           return
// CHECK:         }
}

// -----

func @test_lstm_only_final_step(%X: tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %W: tensor<1x8x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %R: tensor<1x16x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> () {
  %cst = "onnx.NoValue"() {value} : () -> none
  %hn_output, %cf_output = "zhigh.LSTM"(%X, %cst, %cst, %W, %cst, %R, %cst) {direction = "forward", hidden_size = 16 : si64, return_all_steps = 0 : si64} : (tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none, none, tensor<1x8x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, none, tensor<1x16x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, none) -> (tensor<*xf32>, tensor<*xf32>)
  return

// mlir2FileCheck.py -a'["X", "W", "R"]'
// CHECK-LABEL:  func @test_lstm_only_final_step
// CHECK-SAME:   ([[X_:%.+]]: tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, [[W_:%.+]]: tensor<1x8x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, [[R_:%.+]]: tensor<1x16x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[X_]], [[VAR_cst_]], [[VAR_cst_]], [[W_]], [[VAR_cst_]], [[R_]], [[VAR_cst_]]) {direction = "forward", hidden_size = 16 : si64, return_all_steps = 0 : si64} : (tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none, none, tensor<1x8x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, none, tensor<1x16x64xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, none) -> (tensor<1x1x4x16xf32, #zhigh.encoding<{dataLayout = "4DS"}>>, tensor<1x1x4x16xf32, #zhigh.encoding<{dataLayout = "4DS"}>>)
// CHECK:           return
// CHECK:         }
}
