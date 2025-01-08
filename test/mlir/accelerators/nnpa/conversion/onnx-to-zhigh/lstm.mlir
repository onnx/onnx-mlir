// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh --canonicalize %s -split-input-file | FileCheck %s

func.func @test_onnx_to_zhigh_ccfd0(%X: tensor<7x2000x204xf32>, %W: tensor<1x800x204xf32>, %R: tensor<1x800x200xf32>, %B: tensor<1x1600xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %cst, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<7x2000x204xf32>, tensor<1x800x204xf32>, tensor<1x800x200xf32>, tensor<1x1600xf32>, none, none, none, none) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>)
 "func.return"(%Y, %Y_h, %Y_c) : (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>) -> ()

// mlir2FileCheck.py -a'["X", "W", "R", "B"]'
// CHECK-LABEL:  func @test_onnx_to_zhigh_ccfd0
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x800x204xf32>, [[R_:%.+]]: tensor<1x800x200xf32>, [[B_:%.+]]: tensor<1x1600xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>) {
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]]:8 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200, 200, 200]} : (tensor<1x1600xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#2, [[VAR_0_]]#0, [[VAR_0_]]#3, [[VAR_0_]]#1) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#6, [[VAR_0_]]#4, [[VAR_0_]]#7, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x800x204xf32>) -> tensor<1x204x800xf32>
// CHECK:           [[VAR_5_:%.+]]:4 = "onnx.SplitV11"([[VAR_4_]]) {axis = 2 : si64} : (tensor<1x204x800xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.StickForLSTM"([[VAR_5_]]#2, [[VAR_5_]]#0, [[VAR_5_]]#3, [[VAR_5_]]#1) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x800x200xf32>) -> tensor<1x200x800xf32>
// CHECK:           [[VAR_8_:%.+]]:4 = "onnx.SplitV11"([[VAR_7_]]) {axis = 2 : si64} : (tensor<1x200x800xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_9_:%.+]] = "zhigh.StickForLSTM"([[VAR_8_]]#2, [[VAR_8_]]#0, [[VAR_8_]]#3, [[VAR_8_]]#1) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_3_]], [[VAR_cst_]], [[VAR_cst_]], [[VAR_6_]], [[VAR_1_]], [[VAR_9_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, none, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK:           [[VAR_10_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<7x1x2000x200xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Slice"([[VAR_10_]], [[VAR_11_]], [[VAR_14_]], [[VAR_12_]], [[VAR_13_]]) : (tensor<7x1x2000x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.SqueezeV11"([[VAR_15_]]) {axes = [0]} : (tensor<1x1x2000x200xf32>) -> tensor<1x2000x200xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "zhigh.Unstick"([[VAR_cf_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.SqueezeV11"([[VAR_17_]]) {axes = [0]} : (tensor<*xf32>) -> tensor<1x2000x200xf32>
// CHECK:           return [[VAR_10_]], [[VAR_16_]], [[VAR_18_]] : tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_zhigh_ccfd0_reverse(%X: tensor<7x2000x204xf32>, %W: tensor<1x800x204xf32>, %R: tensor<1x800x200xf32>, %B: tensor<1x1600xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %cst, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "reverse", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<7x2000x204xf32>, tensor<1x800x204xf32>, tensor<1x800x200xf32>, tensor<1x1600xf32>, none, none, none, none) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>)
 "func.return"(%Y, %Y_h, %Y_c) : (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_zhigh_ccfd0_reverse
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x800x204xf32>, [[R_:%.+]]: tensor<1x800x200xf32>, [[B_:%.+]]: tensor<1x1600xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>) {
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]]:8 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200, 200, 200]} : (tensor<1x1600xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#2, [[VAR_0_]]#0, [[VAR_0_]]#3, [[VAR_0_]]#1) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#6, [[VAR_0_]]#4, [[VAR_0_]]#7, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x800x204xf32>) -> tensor<1x204x800xf32>
// CHECK:           [[VAR_5_:%.+]]:4 = "onnx.SplitV11"([[VAR_4_]]) {axis = 2 : si64} : (tensor<1x204x800xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.StickForLSTM"([[VAR_5_]]#2, [[VAR_5_]]#0, [[VAR_5_]]#3, [[VAR_5_]]#1) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x800x200xf32>) -> tensor<1x200x800xf32>
// CHECK:           [[VAR_8_:%.+]]:4 = "onnx.SplitV11"([[VAR_7_]]) {axis = 2 : si64} : (tensor<1x200x800xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_9_:%.+]] = "zhigh.StickForLSTM"([[VAR_8_]]#2, [[VAR_8_]]#0, [[VAR_8_]]#3, [[VAR_8_]]#1) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_3_]], [[VAR_cst_]], [[VAR_cst_]], [[VAR_6_]], [[VAR_1_]], [[VAR_9_]], [[VAR_2_]]) {direction = "reverse", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, none, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK:           [[VAR_10_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<7x1x2000x200xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Slice"([[VAR_10_]], [[VAR_11_]], [[VAR_12_]], [[VAR_11_]], [[VAR_12_]]) : (tensor<7x1x2000x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.SqueezeV11"([[VAR_13_]]) {axes = [0]} : (tensor<1x1x2000x200xf32>) -> tensor<1x2000x200xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "zhigh.Unstick"([[VAR_cf_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.SqueezeV11"([[VAR_15_]]) {axes = [0]} : (tensor<*xf32>) -> tensor<1x2000x200xf32>
// CHECK:           return [[VAR_10_]], [[VAR_14_]], [[VAR_16_]] : tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_zhigh_ccfd0_bidir(%X: tensor<7x2000x204xf32>, %W: tensor<2x800x204xf32>, %R: tensor<2x800x200xf32>, %B: tensor<2x1600xf32>) -> (tensor<7x2x2000x200xf32>, tensor<2x2000x200xf32>, tensor<2x2000x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %cst, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "bidirectional", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<7x2000x204xf32>, tensor<2x800x204xf32>, tensor<2x800x200xf32>, tensor<2x1600xf32>, none, none, none, none) -> (tensor<7x2x2000x200xf32>, tensor<2x2000x200xf32>, tensor<2x2000x200xf32>)
 "func.return"(%Y, %Y_h, %Y_c) : (tensor<7x2x2000x200xf32>, tensor<2x2000x200xf32>, tensor<2x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_zhigh_ccfd0_bidir
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<2x800x204xf32>, [[R_:%.+]]: tensor<2x800x200xf32>, [[B_:%.+]]: tensor<2x1600xf32>) -> (tensor<7x2x2000x200xf32>, tensor<2x2000x200xf32>, tensor<2x2000x200xf32>) {
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]]:8 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200, 200, 200]} : (tensor<2x1600xf32>) -> (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#2, [[VAR_0_]]#0, [[VAR_0_]]#3, [[VAR_0_]]#1) : (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#6, [[VAR_0_]]#4, [[VAR_0_]]#7, [[VAR_0_]]#5) : (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<2x800x204xf32>) -> tensor<2x204x800xf32>
// CHECK:           [[VAR_5_:%.+]]:4 = "onnx.SplitV11"([[VAR_4_]]) {axis = 2 : si64} : (tensor<2x204x800xf32>) -> (tensor<2x204x200xf32>, tensor<2x204x200xf32>, tensor<2x204x200xf32>, tensor<2x204x200xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.StickForLSTM"([[VAR_5_]]#2, [[VAR_5_]]#0, [[VAR_5_]]#3, [[VAR_5_]]#1) : (tensor<2x204x200xf32>, tensor<2x204x200xf32>, tensor<2x204x200xf32>, tensor<2x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<2x800x200xf32>) -> tensor<2x200x800xf32>
// CHECK:           [[VAR_8_:%.+]]:4 = "onnx.SplitV11"([[VAR_7_]]) {axis = 2 : si64} : (tensor<2x200x800xf32>) -> (tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>)
// CHECK:           [[VAR_9_:%.+]] = "zhigh.StickForLSTM"([[VAR_8_]]#2, [[VAR_8_]]#0, [[VAR_8_]]#3, [[VAR_8_]]#1) : (tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_3_]], [[VAR_cst_]], [[VAR_cst_]], [[VAR_6_]], [[VAR_1_]], [[VAR_9_]], [[VAR_2_]]) {direction = "bidirectional", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, none, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK:           [[VAR_10_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<7x2x2000x200xf32>
// CHECK-DAG:       [[VAR_15:%.+]]:2 = "onnx.SplitV11"([[VAR_10_]]) {axis = 1 : si64} : (tensor<7x2x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<7x1x2000x200xf32>)
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]]#0, [[VAR_11_]], [[VAR_14_]], [[VAR_12_]], [[VAR_13_]]) : (tensor<7x1x2000x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]]#1, [[VAR_12_]], [[VAR_13_]], [[VAR_12_]], [[VAR_13_]]) : (tensor<7x1x2000x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_16_]], [[VAR_17_]]) {axis = 1 : si64} : (tensor<1x1x2000x200xf32>, tensor<1x1x2000x200xf32>) -> tensor<1x2x2000x200xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.SqueezeV11"([[VAR_18_]]) {axes = [0]} : (tensor<1x2x2000x200xf32>) -> tensor<2x2000x200xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "zhigh.Unstick"([[VAR_cf_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]) {axes = [0]} : (tensor<*xf32>) -> tensor<2x2000x200xf32>
// CHECK:           return [[VAR_10_]], [[VAR_19_]], [[VAR_21_]] : tensor<7x2x2000x200xf32>, tensor<2x2000x200xf32>, tensor<2x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_lstm_in_ccfd1(%X: tensor<7x2000x204xf32>, %W: tensor<1x800x204xf32>, %R: tensor<1x800x200xf32>, %B: tensor<1x1600xf32>, %InitH: tensor<1x2000x200xf32>, %InitC: tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %InitH, %InitC, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<7x2000x204xf32>, tensor<1x800x204xf32>, tensor<1x800x200xf32>, tensor<1x1600xf32>, none, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>, none) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>)
  "func.return"(%Y, %Y_h, %Y_c) : (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_lstm_in_ccfd1
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x800x204xf32>, [[R_:%.+]]: tensor<1x800x200xf32>, [[B_:%.+]]: tensor<1x1600xf32>, [[PARAM_0_:%.+]]: tensor<1x2000x200xf32>, [[PARAM_1_:%.+]]: tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>) {
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_0_:%.+]]:8 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200, 200, 200]} : (tensor<1x1600xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#2, [[VAR_0_]]#0, [[VAR_0_]]#3, [[VAR_0_]]#1) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#6, [[VAR_0_]]#4, [[VAR_0_]]#7, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x800x204xf32>) -> tensor<1x204x800xf32>
// CHECK:           [[VAR_7_:%.+]]:4 = "onnx.SplitV11"([[VAR_6_]]) {axis = 2 : si64} : (tensor<1x204x800xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "zhigh.StickForLSTM"([[VAR_7_]]#2, [[VAR_7_]]#0, [[VAR_7_]]#3, [[VAR_7_]]#1) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x800x200xf32>) -> tensor<1x200x800xf32>
// CHECK:           [[VAR_10_:%.+]]:4 = "onnx.SplitV11"([[VAR_9_]]) {axis = 2 : si64} : (tensor<1x200x800xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_11_:%.+]] = "zhigh.StickForLSTM"([[VAR_10_]]#2, [[VAR_10_]]#0, [[VAR_10_]]#3, [[VAR_10_]]#1) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_1_]], [[VAR_11_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK:           [[VAR_12_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<7x1x2000x200xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_12_]], [[VAR_13_]], [[VAR_16_]], [[VAR_14_]], [[VAR_15_]]) : (tensor<7x1x2000x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.SqueezeV11"([[VAR_17_]]) {axes = [0]} : (tensor<1x1x2000x200xf32>) -> tensor<1x2000x200xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "zhigh.Unstick"([[VAR_cf_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.SqueezeV11"([[VAR_19_]]) {axes = [0]} : (tensor<*xf32>) -> tensor<1x2000x200xf32>
// CHECK:           return [[VAR_12_]], [[VAR_18_]], [[VAR_20_]] : tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_lstm_noY_noYc(%X: tensor<7x2000x204xf32>, %W: tensor<1x800x204xf32>, %R: tensor<1x800x200xf32>, %B: tensor<1x1600xf32>, %InitH: tensor<1x2000x200xf32>, %InitC: tensor<1x2000x200xf32>) -> (tensor<1x2000x200xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %InitH, %InitC, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<7x2000x204xf32>, tensor<1x800x204xf32>, tensor<1x800x200xf32>, tensor<1x1600xf32>, none, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>, none) -> (none, tensor<1x2000x200xf32>, none)
  "func.return"(%Y_h) : (tensor<1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_lstm_noY_noYc
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x800x204xf32>, [[R_:%.+]]: tensor<1x800x200xf32>, [[B_:%.+]]: tensor<1x1600xf32>, [[PARAM_0_:%.+]]: tensor<1x2000x200xf32>, [[PARAM_1_:%.+]]: tensor<1x2000x200xf32>) -> tensor<1x2000x200xf32> {
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK:           [[VAR_0_:%.+]]:8 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200, 200, 200]} : (tensor<1x1600xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#2, [[VAR_0_]]#0, [[VAR_0_]]#3, [[VAR_0_]]#1) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#6, [[VAR_0_]]#4, [[VAR_0_]]#7, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x800x204xf32>) -> tensor<1x204x800xf32>
// CHECK:           [[VAR_7_:%.+]]:4 = "onnx.SplitV11"([[VAR_6_]]) {axis = 2 : si64} : (tensor<1x204x800xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "zhigh.StickForLSTM"([[VAR_7_]]#2, [[VAR_7_]]#0, [[VAR_7_]]#3, [[VAR_7_]]#1) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x800x200xf32>) -> tensor<1x200x800xf32>
// CHECK:           [[VAR_10_:%.+]]:4 = "onnx.SplitV11"([[VAR_9_]]) {axis = 2 : si64} : (tensor<1x200x800xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_11_:%.+]] = "zhigh.StickForLSTM"([[VAR_10_]]#2, [[VAR_10_]]#0, [[VAR_10_]]#3, [[VAR_10_]]#1) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_1_]], [[VAR_11_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = 1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK-DAG:       [[VAR_12_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_12_]], [[VAR_13_]], [[VAR_16_]], [[VAR_14_]], [[VAR_15_]]) : (tensor<*xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.SqueezeV11"([[VAR_17_]]) {axes = [0]} : (tensor<1x1x2000x200xf32>) -> tensor<1x2000x200xf32>
// CHECK:           return [[VAR_18_]] : tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_lstm_noYh(%X: tensor<7x2000x204xf32>, %W: tensor<1x800x204xf32>, %R: tensor<1x800x200xf32>, %B: tensor<1x1600xf32>, %InitH: tensor<1x2000x200xf32>, %InitC: tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %InitH, %InitC, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<7x2000x204xf32>, tensor<1x800x204xf32>, tensor<1x800x200xf32>, tensor<1x1600xf32>, none, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>, none) -> (tensor<7x1x2000x200xf32>, none, tensor<1x2000x200xf32>)
  "func.return"(%Y, %Y_c) : (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_lstm_noYh
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x800x204xf32>, [[R_:%.+]]: tensor<1x800x200xf32>, [[B_:%.+]]: tensor<1x1600xf32>, [[PARAM_0_:%.+]]: tensor<1x2000x200xf32>, [[PARAM_1_:%.+]]: tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) {
// CHECK:           [[VAR_0_:%.+]]:8 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200, 200, 200]} : (tensor<1x1600xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#2, [[VAR_0_]]#0, [[VAR_0_]]#3, [[VAR_0_]]#1) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#6, [[VAR_0_]]#4, [[VAR_0_]]#7, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x800x204xf32>) -> tensor<1x204x800xf32>
// CHECK:           [[VAR_7_:%.+]]:4 = "onnx.SplitV11"([[VAR_6_]]) {axis = 2 : si64} : (tensor<1x204x800xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "zhigh.StickForLSTM"([[VAR_7_]]#2, [[VAR_7_]]#0, [[VAR_7_]]#3, [[VAR_7_]]#1) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x800x200xf32>) -> tensor<1x200x800xf32>
// CHECK:           [[VAR_10_:%.+]]:4 = "onnx.SplitV11"([[VAR_9_]]) {axis = 2 : si64} : (tensor<1x200x800xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_11_:%.+]] = "zhigh.StickForLSTM"([[VAR_10_]]#2, [[VAR_10_]]#0, [[VAR_10_]]#3, [[VAR_10_]]#1) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_1_]], [[VAR_11_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK:           [[VAR_12_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<7x1x2000x200xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "zhigh.Unstick"([[VAR_cf_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.SqueezeV11"([[VAR_13_]]) {axes = [0]} : (tensor<*xf32>) -> tensor<1x2000x200xf32>
// CHECK:           return [[VAR_12_]], [[VAR_14_]] : tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_lstm_noB_noY_noYc(%X: tensor<7x2000x204xf32>, %W: tensor<1x800x204xf32>, %R: tensor<1x800x200xf32>, %InitH: tensor<1x2000x200xf32>, %InitC: tensor<1x2000x200xf32>) -> (tensor<1x2000x200xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %cst, %cst, %InitH, %InitC, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<7x2000x204xf32>, tensor<1x800x204xf32>, tensor<1x800x200xf32>, none, none, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>, none) -> (none, tensor<1x2000x200xf32>, none)
  "func.return"(%Y_h) : (tensor<1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_lstm_noB_noY_noYc
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x800x204xf32>, [[R_:%.+]]: tensor<1x800x200xf32>, [[PARAM_0_:%.+]]: tensor<1x2000x200xf32>, [[PARAM_1_:%.+]]: tensor<1x2000x200xf32>) -> tensor<1x2000x200xf32> {
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-DAG:       [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x800x204xf32>) -> tensor<1x204x800xf32>
// CHECK:           [[VAR_4_:%.+]]:4 = "onnx.SplitV11"([[VAR_3_]]) {axis = 2 : si64} : (tensor<1x204x800xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.StickForLSTM"([[VAR_4_]]#2, [[VAR_4_]]#0, [[VAR_4_]]#3, [[VAR_4_]]#1) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x800x200xf32>) -> tensor<1x200x800xf32>
// CHECK:           [[VAR_7_:%.+]]:4 = "onnx.SplitV11"([[VAR_6_]]) {axis = 2 : si64} : (tensor<1x200x800xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_8_:%.+]] = "zhigh.StickForLSTM"([[VAR_7_]]#2, [[VAR_7_]]#0, [[VAR_7_]]#3, [[VAR_7_]]#1) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_5_]], [[CST]], [[VAR_8_]], [[CST]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = 1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, none, tensor<*xf16>, none) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK-DAG:       [[VAR_9_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Slice"([[VAR_9_]], [[VAR_10_]], [[VAR_13_]], [[VAR_11_]], [[VAR_12_]]) : (tensor<*xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.SqueezeV11"([[VAR_14_]]) {axes = [0]} : (tensor<1x1x2000x200xf32>) -> tensor<1x2000x200xf32>
// CHECK:           return [[VAR_15_]] : tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_lstm_noB_noYh(%X: tensor<7x2000x204xf32>, %W: tensor<1x800x204xf32>, %R: tensor<1x800x200xf32>, %InitH: tensor<1x2000x200xf32>, %InitC: tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %cst, %cst, %InitH, %InitC, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<7x2000x204xf32>, tensor<1x800x204xf32>, tensor<1x800x200xf32>, none, none, tensor<1x2000x200xf32>, tensor<1x2000x200xf32>, none) -> (tensor<7x1x2000x200xf32>, none, tensor<1x2000x200xf32>)
  "func.return"(%Y, %Y_c) : (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_lstm_noB_noYh
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x800x204xf32>, [[R_:%.+]]: tensor<1x800x200xf32>, [[PARAM_0_:%.+]]: tensor<1x2000x200xf32>, [[PARAM_1_:%.+]]: tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) {
// CHECK-DAG:       [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x800x204xf32>) -> tensor<1x204x800xf32>
// CHECK:           [[VAR_4_:%.+]]:4 = "onnx.SplitV11"([[VAR_3_]]) {axis = 2 : si64} : (tensor<1x204x800xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.StickForLSTM"([[VAR_4_]]#2, [[VAR_4_]]#0, [[VAR_4_]]#3, [[VAR_4_]]#1) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x800x200xf32>) -> tensor<1x200x800xf32>
// CHECK:           [[VAR_7_:%.+]]:4 = "onnx.SplitV11"([[VAR_6_]]) {axis = 2 : si64} : (tensor<1x200x800xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_8_:%.+]] = "zhigh.StickForLSTM"([[VAR_7_]]#2, [[VAR_7_]]#0, [[VAR_7_]]#3, [[VAR_7_]]#1) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_5_]], [[CST]], [[VAR_8_]], [[CST]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, none, tensor<*xf16>, none) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK:           [[VAR_9_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<7x1x2000x200xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "zhigh.Unstick"([[VAR_cf_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[VAR_10_]]) {axes = [0]} : (tensor<*xf32>) -> tensor<1x2000x200xf32>
// CHECK:           return [[VAR_9_]], [[VAR_11_]] : tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_zhigh_ccfd0_dyn(%X: tensor<?x?x?xf32>, %W: tensor<1x800x?xf32>, %R: tensor<1x800x200xf32>, %B: tensor<1x1600xf32>) -> (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %cst, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<?x?x?xf32>, tensor<1x800x?xf32>, tensor<1x800x200xf32>, tensor<1x1600xf32>, none, none, none, none) -> (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>)
 "func.return"(%Y, %Y_h, %Y_c) : (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_zhigh_ccfd0_dyn
// CHECK-SAME:   ([[X_:%.+]]: tensor<?x?x?xf32>, [[W_:%.+]]: tensor<1x800x?xf32>, [[R_:%.+]]: tensor<1x800x200xf32>, [[B_:%.+]]: tensor<1x1600xf32>) -> (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>) {
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]]:8 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200, 200, 200]} : (tensor<1x1600xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#2, [[VAR_0_]]#0, [[VAR_0_]]#3, [[VAR_0_]]#1) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#6, [[VAR_0_]]#4, [[VAR_0_]]#7, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x800x?xf32>) -> tensor<1x?x800xf32>
// CHECK:           [[VAR_5_:%.+]]:4 = "onnx.SplitV11"([[VAR_4_]]) {axis = 2 : si64} : (tensor<1x?x800xf32>) -> (tensor<1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.StickForLSTM"([[VAR_5_]]#2, [[VAR_5_]]#0, [[VAR_5_]]#3, [[VAR_5_]]#1) : (tensor<1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x800x200xf32>) -> tensor<1x200x800xf32>
// CHECK:           [[VAR_8_:%.+]]:4 = "onnx.SplitV11"([[VAR_7_]]) {axis = 2 : si64} : (tensor<1x200x800xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_9_:%.+]] = "zhigh.StickForLSTM"([[VAR_8_]]#2, [[VAR_8_]]#0, [[VAR_8_]]#3, [[VAR_8_]]#1) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_3_]], [[VAR_cst_]], [[VAR_cst_]], [[VAR_6_]], [[VAR_1_]], [[VAR_9_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, none, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK:           [[VAR_10_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<?x1x?x200xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Slice"([[VAR_10_]], [[VAR_11_]], [[VAR_14_]], [[VAR_12_]], [[VAR_13_]]) : (tensor<?x1x?x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x?x200xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.SqueezeV11"([[VAR_15_]]) {axes = [0]} : (tensor<1x1x?x200xf32>) -> tensor<1x?x200xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "zhigh.Unstick"([[VAR_cf_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.SqueezeV11"([[VAR_17_]]) {axes = [0]} : (tensor<*xf32>) -> tensor<1x?x200xf32>
// CHECK:           return [[VAR_10_]], [[VAR_16_]], [[VAR_18_]] : tensor<?x1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_zhigh_ccfd1_dyn(%X: tensor<?x?x?xf32>, %W: tensor<1x800x?xf32>, %R: tensor<1x800x200xf32>, %B: tensor<1x1600xf32>, %InitH: tensor<1x?x200xf32>, %InitC: tensor<1x?x200xf32>) -> (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %InitH, %InitC, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<?x?x?xf32>, tensor<1x800x?xf32>, tensor<1x800x200xf32>, tensor<1x1600xf32>, none, tensor<1x?x200xf32>, tensor<1x?x200xf32>, none) -> (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>)
 "func.return"(%Y, %Y_h, %Y_c) : (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_zhigh_ccfd1_dyn
// CHECK-SAME:   ([[X_:%.+]]: tensor<?x?x?xf32>, [[W_:%.+]]: tensor<1x800x?xf32>, [[R_:%.+]]: tensor<1x800x200xf32>, [[B_:%.+]]: tensor<1x1600xf32>, [[PARAM_0_:%.+]]: tensor<1x?x200xf32>, [[PARAM_1_:%.+]]: tensor<1x?x200xf32>) -> (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>) {
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_0_:%.+]]:8 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200, 200, 200]} : (tensor<1x1600xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#2, [[VAR_0_]]#0, [[VAR_0_]]#3, [[VAR_0_]]#1) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#6, [[VAR_0_]]#4, [[VAR_0_]]#7, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x?x200xf32>) -> tensor<1x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<1x?x200xf32>) -> tensor<1x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x800x?xf32>) -> tensor<1x?x800xf32>
// CHECK:           [[VAR_7_:%.+]]:4 = "onnx.SplitV11"([[VAR_6_]]) {axis = 2 : si64} : (tensor<1x?x800xf32>) -> (tensor<1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "zhigh.StickForLSTM"([[VAR_7_]]#2, [[VAR_7_]]#0, [[VAR_7_]]#3, [[VAR_7_]]#1) : (tensor<1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x800x200xf32>) -> tensor<1x200x800xf32>
// CHECK:           [[VAR_10_:%.+]]:4 = "onnx.SplitV11"([[VAR_9_]]) {axis = 2 : si64} : (tensor<1x200x800xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_11_:%.+]] = "zhigh.StickForLSTM"([[VAR_10_]]#2, [[VAR_10_]]#0, [[VAR_10_]]#3, [[VAR_10_]]#1) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_1_]], [[VAR_11_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK:           [[VAR_12_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<?x1x?x200xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_12_]], [[VAR_13_]], [[VAR_16_]], [[VAR_14_]], [[VAR_15_]]) : (tensor<?x1x?x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x?x200xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.SqueezeV11"([[VAR_17_]]) {axes = [0]} : (tensor<1x1x?x200xf32>) -> tensor<1x?x200xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "zhigh.Unstick"([[VAR_cf_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK:           return [[VAR_12_]], [[VAR_18_]], [[VAR_20_]] : tensor<?x1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_zhigh_ccfd0_bidir_dyn(%X: tensor<?x?x?xf32>, %W: tensor<2x800x?xf32>, %R: tensor<2x800x200xf32>, %B: tensor<2x1600xf32>) -> (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %cst, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "bidirectional", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<?x?x?xf32>, tensor<2x800x?xf32>, tensor<2x800x200xf32>, tensor<2x1600xf32>, none, none, none, none) -> (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>)
 "func.return"(%Y, %Y_h, %Y_c) : (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_zhigh_ccfd0_bidir_dyn
// CHECK-SAME:   ([[X_:%.+]]: tensor<?x?x?xf32>, [[W_:%.+]]: tensor<2x800x?xf32>, [[R_:%.+]]: tensor<2x800x200xf32>, [[B_:%.+]]: tensor<2x1600xf32>) -> (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>) {
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]]:8 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200, 200, 200]} : (tensor<2x1600xf32>) -> (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#2, [[VAR_0_]]#0, [[VAR_0_]]#3, [[VAR_0_]]#1) : (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#6, [[VAR_0_]]#4, [[VAR_0_]]#7, [[VAR_0_]]#5) : (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<2x800x?xf32>) -> tensor<2x?x800xf32>
// CHECK:           [[VAR_5_:%.+]]:4 = "onnx.SplitV11"([[VAR_4_]]) {axis = 2 : si64} : (tensor<2x?x800xf32>) -> (tensor<2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.StickForLSTM"([[VAR_5_]]#2, [[VAR_5_]]#0, [[VAR_5_]]#3, [[VAR_5_]]#1) : (tensor<2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<2x800x200xf32>) -> tensor<2x200x800xf32>
// CHECK:           [[VAR_8_:%.+]]:4 = "onnx.SplitV11"([[VAR_7_]]) {axis = 2 : si64} : (tensor<2x200x800xf32>) -> (tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>)
// CHECK:           [[VAR_9_:%.+]] = "zhigh.StickForLSTM"([[VAR_8_]]#2, [[VAR_8_]]#0, [[VAR_8_]]#3, [[VAR_8_]]#1) : (tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_3_]], [[VAR_cst_]], [[VAR_cst_]], [[VAR_6_]], [[VAR_1_]], [[VAR_9_]], [[VAR_2_]]) {direction = "bidirectional", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, none, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK:           [[VAR_10_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<?x2x?x200xf32>
// CHECK-DAG:       [[VAR_15_:%.+]]:2 = "onnx.SplitV11"([[VAR_10_]]) {axis = 1 : si64} : (tensor<?x2x?x200xf32>) -> (tensor<?x1x?x200xf32>, tensor<?x1x?x200xf32>)
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]]#0, [[VAR_11_]], [[VAR_14_]], [[VAR_12_]], [[VAR_13_]]) : (tensor<?x1x?x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x?x200xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]]#1, [[VAR_12_]], [[VAR_13_]], [[VAR_12_]], [[VAR_13_]]) : (tensor<?x1x?x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x?x200xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_16_]], [[VAR_17_]]) {axis = 1 : si64} : (tensor<1x1x?x200xf32>, tensor<1x1x?x200xf32>) -> tensor<1x2x?x200xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.SqueezeV11"([[VAR_18_]]) {axes = [0]} : (tensor<1x2x?x200xf32>) -> tensor<2x?x200xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "zhigh.Unstick"([[VAR_cf_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]) {axes = [0]} : (tensor<*xf32>) -> tensor<2x?x200xf32>
// CHECK:           return [[VAR_10_]], [[VAR_19_]], [[VAR_21_]] : tensor<?x2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_zhigh_ccfd1_bidir_dyn(%X: tensor<?x?x?xf32>, %W: tensor<2x800x?xf32>, %R: tensor<2x800x200xf32>, %B: tensor<2x1600xf32>, %InitH: tensor<2x?x200xf32>, %InitC: tensor<2x?x200xf32>) -> (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %InitH, %InitC, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "bidirectional", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<?x?x?xf32>, tensor<2x800x?xf32>, tensor<2x800x200xf32>, tensor<2x1600xf32>, none, tensor<2x?x200xf32>, tensor<2x?x200xf32>, none) -> (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>)
 "func.return"(%Y, %Y_h, %Y_c) : (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_zhigh_ccfd1_bidir_dyn
// CHECK-SAME:   ([[X_:%.+]]: tensor<?x?x?xf32>, [[W_:%.+]]: tensor<2x800x?xf32>, [[R_:%.+]]: tensor<2x800x200xf32>, [[B_:%.+]]: tensor<2x1600xf32>, [[PARAM_0_:%.+]]: tensor<2x?x200xf32>, [[PARAM_1_:%.+]]: tensor<2x?x200xf32>) -> (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>) {
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_0_:%.+]]:8 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200, 200, 200]} : (tensor<2x1600xf32>) -> (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>)
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#2, [[VAR_0_]]#0, [[VAR_0_]]#3, [[VAR_0_]]#1) : (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForLSTM"([[VAR_0_]]#6, [[VAR_0_]]#4, [[VAR_0_]]#7, [[VAR_0_]]#5) : (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<2x?x200xf32>) -> tensor<2x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<2x?x200xf32>) -> tensor<2x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<2x800x?xf32>) -> tensor<2x?x800xf32>
// CHECK:           [[VAR_7_:%.+]]:4 = "onnx.SplitV11"([[VAR_6_]]) {axis = 2 : si64} : (tensor<2x?x800xf32>) -> (tensor<2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "zhigh.StickForLSTM"([[VAR_7_]]#2, [[VAR_7_]]#0, [[VAR_7_]]#3, [[VAR_7_]]#1) : (tensor<2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<2x800x200xf32>) -> tensor<2x200x800xf32>
// CHECK:           [[VAR_10_:%.+]]:4 = "onnx.SplitV11"([[VAR_9_]]) {axis = 2 : si64} : (tensor<2x200x800xf32>) -> (tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>)
// CHECK:           [[VAR_11_:%.+]] = "zhigh.StickForLSTM"([[VAR_10_]]#2, [[VAR_10_]]#0, [[VAR_10_]]#3, [[VAR_10_]]#1) : (tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[hn_output_:%.+]], [[VAR_cf_output_:%.+]] = "zhigh.LSTM"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_1_]], [[VAR_11_]], [[VAR_2_]]) {direction = "bidirectional", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> (tensor<*xf16>, tensor<*xf16>)
// CHECK:           [[VAR_12_:%.+]] = "zhigh.Unstick"([[hn_output_]]) : (tensor<*xf16>) -> tensor<?x2x?x200xf32>
// CHECK-DAG:       [[VAR_17_:%.+]]:2 = "onnx.SplitV11"([[VAR_12_]]) {axis = 1 : si64} : (tensor<?x2x?x200xf32>) -> (tensor<?x1x?x200xf32>, tensor<?x1x?x200xf32>)
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_17_]]#0, [[VAR_13_]], [[VAR_16_]], [[VAR_14_]], [[VAR_15_]]) : (tensor<?x1x?x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x?x200xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_17_]]#1, [[VAR_14_]], [[VAR_15_]], [[VAR_14_]], [[VAR_15_]]) : (tensor<?x1x?x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x?x200xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Concat"([[VAR_18_]], [[VAR_19_]]) {axis = 1 : si64} : (tensor<1x1x?x200xf32>, tensor<1x1x?x200xf32>) -> tensor<1x2x?x200xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]) {axes = [0]} : (tensor<1x2x?x200xf32>) -> tensor<2x?x200xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "zhigh.Unstick"([[VAR_cf_output_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.SqueezeV11"([[VAR_22_]]) {axes = [0]} : (tensor<*xf32>) -> tensor<2x?x200xf32>
// CHECK:           return [[VAR_12_]], [[VAR_21_]], [[VAR_23_]] : tensor<?x2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>
// CHECK:         }
}
// -----

// COM : Maximum hidden_size in LSTM is 8192. Not lowered when using 8193.

func.func @test_onnx_to_zhigh_lstm_exceed_num_hidden(%X: tensor<7x2000x204xf32>, %W: tensor<1x16384x204xf32>, %R: tensor<1x16384x8193xf32>, %B: tensor<1x16386xf32>) -> (tensor<7x1x2000x8193xf32>, tensor<1x2000x8193xf32>, tensor<1x2000x8193xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %cst, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 8193 : si64, onnx_node_name = "lstm" } : (tensor<7x2000x204xf32>, tensor<1x16384x204xf32>, tensor<1x16384x8193xf32>, tensor<1x16386xf32>, none, none, none, none) -> (tensor<7x1x2000x8193xf32>, tensor<1x2000x8193xf32>, tensor<1x2000x8193xf32>)
 "func.return"(%Y, %Y_h, %Y_c) : (tensor<7x1x2000x8193xf32>, tensor<1x2000x8193xf32>, tensor<1x2000x8193xf32>) -> ()

  // CHECK-LABEL: test_onnx_to_zhigh_lstm_exceed_num_hidden
  // CHECK: "onnx.LSTM"
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func.func @test_exceed_limit_lstm(%X: tensor<7x32769x204xf32>, %W: tensor<1x800x204xf32>, %R: tensor<1x800x200xf32>, %B: tensor<1x1600xf32>) -> (tensor<7x1x32769x200xf32>, tensor<1x32769x200xf32>, tensor<1x32769x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %cst, %cst, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, onnx_node_name = "lstm" } : (tensor<7x32769x204xf32>, tensor<1x800x204xf32>, tensor<1x800x200xf32>, tensor<1x1600xf32>, none, none, none, none) -> (tensor<7x1x32769x200xf32>, tensor<1x32769x200xf32>, tensor<1x32769x200xf32>)
 "func.return"(%Y, %Y_h, %Y_c) : (tensor<7x1x32769x200xf32>, tensor<1x32769x200xf32>, tensor<1x32769x200xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_lstm
// CHECK:        "onnx.LSTM"
}
