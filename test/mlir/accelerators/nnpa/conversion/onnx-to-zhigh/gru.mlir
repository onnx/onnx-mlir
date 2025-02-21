// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh --canonicalize %s -split-input-file | FileCheck %s

func.func @test_onnx_to_zhigh_gru0(%X: tensor<7x2000x204xf32>, %W: tensor<1x600x204xf32>, %R: tensor<1x600x200xf32>, %B: tensor<1x1200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h = "onnx.GRU"(%X, %W, %R, %B, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, linear_before_reset = 1 : si64, onnx_node_name = "gru" } : (tensor<7x2000x204xf32>, tensor<1x600x204xf32>, tensor<1x600x200xf32>, tensor<1x1200xf32>, none, none) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>)
 "func.return"(%Y, %Y_h) : (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) -> ()

// mlir2FileCheck.py -a'["X", "W", "R", "B"]'
// CHECK-LABEL:  func @test_onnx_to_zhigh_gru0
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x600x204xf32>, [[R_:%.+]]: tensor<1x600x200xf32>, [[B_:%.+]]: tensor<1x1200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) {
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]]:6 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200]} : (tensor<1x1200xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#0, [[VAR_0_]]#1, [[VAR_0_]]#2) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#3, [[VAR_0_]]#4, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x600x204xf32>) -> tensor<1x204x600xf32>
// CHECK:           [[VAR_5_:%.+]]:3 = "onnx.SplitV11"([[VAR_4_]]) {axis = 2 : si64} : (tensor<1x204x600xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.StickForGRU"([[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x600x200xf32>) -> tensor<1x200x600xf32>
// CHECK:           [[VAR_8_:%.+]]:3 = "onnx.SplitV11"([[VAR_7_]]) {axis = 2 : si64} : (tensor<1x200x600xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_9_:%.+]] = "zhigh.StickForGRU"([[VAR_8_]]#0, [[VAR_8_]]#1, [[VAR_8_]]#2) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[VAR_10_:%.+]] = "zhigh.GRU"([[VAR_3_]], [[VAR_cst_]], [[VAR_6_]], [[VAR_1_]], [[VAR_9_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.Unstick"([[VAR_10_]]) : (tensor<*xf16>) -> tensor<7x1x2000x200xf32>
// CHECK:           [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_11_]], [[VAR_12_]], [[VAR_15_]], [[VAR_13_]], [[VAR_14_]]) : (tensor<7x1x2000x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.SqueezeV11"([[VAR_16_]]) {axes = [0]} : (tensor<1x1x2000x200xf32>) -> tensor<1x2000x200xf32>
// CHECK:           return [[VAR_11_]], [[VAR_17_]] : tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_gru1(%X: tensor<7x2000x204xf32>, %W: tensor<1x600x204xf32>, %R: tensor<1x600x200xf32>, %B: tensor<1x1200xf32>, %InitH: tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%X, %W, %R, %B, %cst, %InitH) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, linear_before_reset = 1 : si64, onnx_node_name = "gru" } : (tensor<7x2000x204xf32>, tensor<1x600x204xf32>, tensor<1x600x200xf32>, tensor<1x1200xf32>, none, tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>)
  "func.return"(%Y, %Y_h) : (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_gru1
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x600x204xf32>, [[R_:%.+]]: tensor<1x600x200xf32>, [[B_:%.+]]: tensor<1x1200xf32>, [[PARAM_0_:%.+]]: tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>) {
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK:           [[VAR_0_:%.+]]:6 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200]} : (tensor<1x1200xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#0, [[VAR_0_]]#1, [[VAR_0_]]#2) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#3, [[VAR_0_]]#4, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x600x204xf32>) -> tensor<1x204x600xf32>
// CHECK:           [[VAR_6_:%.+]]:3 = "onnx.SplitV11"([[VAR_5_]]) {axis = 2 : si64} : (tensor<1x204x600xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_7_:%.+]] = "zhigh.StickForGRU"([[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x600x200xf32>) -> tensor<1x200x600xf32>
// CHECK:           [[VAR_9_:%.+]]:3 = "onnx.SplitV11"([[VAR_8_]]) {axis = 2 : si64} : (tensor<1x200x600xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_10_:%.+]] = "zhigh.StickForGRU"([[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.GRU"([[VAR_3_]], [[VAR_4_]], [[VAR_7_]], [[VAR_1_]], [[VAR_10_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16>
// CHECK:           [[VAR_12_:%.+]] = "zhigh.Unstick"([[VAR_11_]]) : (tensor<*xf16>) -> tensor<7x1x2000x200xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_12_]], [[VAR_13_]], [[VAR_16_]], [[VAR_14_]], [[VAR_15_]]) : (tensor<7x1x2000x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.SqueezeV11"([[VAR_17_]]) {axes = [0]} : (tensor<1x1x2000x200xf32>) -> tensor<1x2000x200xf32>
// CHECK:           return [[VAR_12_]], [[VAR_18_]] : tensor<7x1x2000x200xf32>, tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_gru_noY_noYc(%X: tensor<7x2000x204xf32>, %W: tensor<1x600x204xf32>, %R: tensor<1x600x200xf32>, %B: tensor<1x1200xf32>, %InitH: tensor<1x2000x200xf32>) -> (tensor<1x2000x200xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%X, %W, %R, %B, %cst, %InitH) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, linear_before_reset = 1 : si64, onnx_node_name = "gru" } : (tensor<7x2000x204xf32>, tensor<1x600x204xf32>, tensor<1x600x200xf32>, tensor<1x1200xf32>, none, tensor<1x2000x200xf32>) -> (none, tensor<1x2000x200xf32>)
  "func.return"(%Y_h) : (tensor<1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_gru_noY_noYc
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x600x204xf32>, [[R_:%.+]]: tensor<1x600x200xf32>, [[B_:%.+]]: tensor<1x1200xf32>, [[PARAM_0_:%.+]]: tensor<1x2000x200xf32>) -> tensor<1x2000x200xf32> {
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK:           [[VAR_0_:%.+]]:6 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200]} : (tensor<1x1200xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#0, [[VAR_0_]]#1, [[VAR_0_]]#2) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#3, [[VAR_0_]]#4, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x600x204xf32>) -> tensor<1x204x600xf32>
// CHECK:           [[VAR_6_:%.+]]:3 = "onnx.SplitV11"([[VAR_5_]]) {axis = 2 : si64} : (tensor<1x204x600xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_7_:%.+]] = "zhigh.StickForGRU"([[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x600x200xf32>) -> tensor<1x200x600xf32>
// CHECK:           [[VAR_9_:%.+]]:3 = "onnx.SplitV11"([[VAR_8_]]) {axis = 2 : si64} : (tensor<1x200x600xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_10_:%.+]] = "zhigh.StickForGRU"([[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.GRU"([[VAR_3_]], [[VAR_4_]], [[VAR_7_]], [[VAR_1_]], [[VAR_10_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = 1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_12_:%.+]] = "zhigh.Unstick"([[VAR_11_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_12_]], [[VAR_13_]], [[VAR_16_]], [[VAR_14_]], [[VAR_15_]]) : (tensor<*xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.SqueezeV11"([[VAR_17_]]) {axes = [0]} : (tensor<1x1x2000x200xf32>) -> tensor<1x2000x200xf32
// CHECK:           return [[VAR_18_]] : tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_gru_noYh(%X: tensor<7x2000x204xf32>, %W: tensor<1x600x204xf32>, %R: tensor<1x600x200xf32>, %B: tensor<1x1200xf32>, %InitH: tensor<1x2000x200xf32>, %InitC: tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%X, %W, %R, %B, %cst, %InitH) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, linear_before_reset = 1 : si64, onnx_node_name = "gru" } : (tensor<7x2000x204xf32>, tensor<1x600x204xf32>, tensor<1x600x200xf32>, tensor<1x1200xf32>, none, tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, none)
  "func.return"(%Y) : (tensor<7x1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_gru_noYh
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x600x204xf32>, [[R_:%.+]]: tensor<1x600x200xf32>, [[B_:%.+]]: tensor<1x1200xf32>, [[PARAM_0_:%.+]]: tensor<1x2000x200xf32>, [[PARAM_1_:%.+]]: tensor<1x2000x200xf32>) -> tensor<7x1x2000x200xf32> {
// CHECK:           [[VAR_0_:%.+]]:6 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200]} : (tensor<1x1200xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#0, [[VAR_0_]]#1, [[VAR_0_]]#2) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#3, [[VAR_0_]]#4, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x600x204xf32>) -> tensor<1x204x600xf32>
// CHECK:           [[VAR_6_:%.+]]:3 = "onnx.SplitV11"([[VAR_5_]]) {axis = 2 : si64} : (tensor<1x204x600xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_7_:%.+]] = "zhigh.StickForGRU"([[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x600x200xf32>) -> tensor<1x200x600xf32>
// CHECK:           [[VAR_9_:%.+]]:3 = "onnx.SplitV11"([[VAR_8_]]) {axis = 2 : si64} : (tensor<1x200x600xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_10_:%.+]] = "zhigh.StickForGRU"([[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.GRU"([[VAR_3_]], [[VAR_4_]], [[VAR_7_]], [[VAR_1_]], [[VAR_10_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16>
// CHECK:           [[VAR_12_:%.+]] = "zhigh.Unstick"([[VAR_11_]]) : (tensor<*xf16>) -> tensor<7x1x2000x200xf32>
// CHECK:           return [[VAR_12_]] : tensor<7x1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_gru_noB_noY_noYc(%X: tensor<7x2000x204xf32>, %W: tensor<1x600x204xf32>, %R: tensor<1x600x200xf32>, %InitH: tensor<1x2000x200xf32>) -> (tensor<1x2000x200xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%X, %W, %R, %cst, %cst, %InitH) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, linear_before_reset = 1 : si64, onnx_node_name = "gru" } : (tensor<7x2000x204xf32>, tensor<1x600x204xf32>, tensor<1x600x200xf32>, none, none, tensor<1x2000x200xf32>) -> (none, tensor<1x2000x200xf32>)
  "func.return"(%Y_h) : (tensor<1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_gru_noB_noY_noYc
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x600x204xf32>, [[R_:%.+]]: tensor<1x600x200xf32>, [[PARAM_0_:%.+]]: tensor<1x2000x200xf32>) -> tensor<1x2000x200xf32> {
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-DAG:       [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x600x204xf32>) -> tensor<1x204x600xf32>
// CHECK:           [[VAR_3_:%.+]]:3 = "onnx.SplitV11"([[VAR_2_]]) {axis = 2 : si64} : (tensor<1x204x600xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.StickForGRU"([[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x600x200xf32>) -> tensor<1x200x600xf32>
// CHECK:           [[VAR_6_:%.+]]:3 = "onnx.SplitV11"([[VAR_5_]]) {axis = 2 : si64} : (tensor<1x200x600xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_7_:%.+]] = "zhigh.StickForGRU"([[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.GRU"([[VAR_0_]], [[VAR_1_]], [[VAR_4_]], [[CST]], [[VAR_7_]], [[CST]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = 1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, none, tensor<*xf16>, none) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_9_:%.+]] = "zhigh.Unstick"([[VAR_8_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Slice"([[VAR_9_]], [[VAR_10_]], [[VAR_13_]], [[VAR_11_]], [[VAR_12_]]) : (tensor<*xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x2000x200xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.SqueezeV11"([[VAR_14_]]) {axes = [0]} : (tensor<1x1x2000x200xf32>) -> tensor<1x2000x200xf32
// CHECK:           return [[VAR_15_]] : tensor<1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_gru_noB_noYh(%X: tensor<7x2000x204xf32>, %W: tensor<1x600x204xf32>, %R: tensor<1x600x200xf32>, %InitH: tensor<1x2000x200xf32>, %InitC: tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%X, %W, %R, %cst, %cst, %InitH) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, linear_before_reset = 1 : si64, onnx_node_name = "gru" } : (tensor<7x2000x204xf32>, tensor<1x600x204xf32>, tensor<1x600x200xf32>, none, none, tensor<1x2000x200xf32>) -> (tensor<7x1x2000x200xf32>, none)
  "func.return"(%Y) : (tensor<7x1x2000x200xf32>) -> ()

// CHECK-LABEL:  func @test_gru_noB_noYh
// CHECK-SAME:   ([[X_:%.+]]: tensor<7x2000x204xf32>, [[W_:%.+]]: tensor<1x600x204xf32>, [[R_:%.+]]: tensor<1x600x200xf32>, [[PARAM_0_:%.+]]: tensor<1x2000x200xf32>, [[PARAM_1_:%.+]]: tensor<1x2000x200xf32>) -> tensor<7x1x2000x200xf32> {
// CHECK-DAG:       [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<7x2000x204xf32>) -> tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2000x200xf32>) -> tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x600x204xf32>) -> tensor<1x204x600xf32>
// CHECK:           [[VAR_3_:%.+]]:3 = "onnx.SplitV11"([[VAR_2_]]) {axis = 2 : si64} : (tensor<1x204x600xf32>) -> (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>)
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.StickForGRU"([[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2) : (tensor<1x204x200xf32>, tensor<1x204x200xf32>, tensor<1x204x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x600x200xf32>) -> tensor<1x200x600xf32>
// CHECK:           [[VAR_6_:%.+]]:3 = "onnx.SplitV11"([[VAR_5_]]) {axis = 2 : si64} : (tensor<1x200x600xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_7_:%.+]] = "zhigh.StickForGRU"([[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.GRU"([[VAR_0_]], [[VAR_1_]], [[VAR_4_]], [[CST]], [[VAR_7_]], [[CST]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<7x2000x204xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x2000x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<*xf16>, none, tensor<*xf16>, none) -> tensor<*xf16>
// CHECK:           [[VAR_9_:%.+]] = "zhigh.Unstick"([[VAR_8_]]) : (tensor<*xf16>) -> tensor<7x1x2000x200xf32>
// CHECK:           return [[VAR_9_]] : tensor<7x1x2000x200xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_zhigh_gru0_dyn(%X: tensor<?x?x?xf32>, %W: tensor<1x600x?xf32>, %R: tensor<1x600x200xf32>, %B: tensor<1x1200xf32>) -> (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h = "onnx.GRU"(%X, %W, %R, %B, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64, linear_before_reset = 1 : si64, onnx_node_name = "gru" } : (tensor<?x?x?xf32>, tensor<1x600x?xf32>, tensor<1x600x200xf32>, tensor<1x1200xf32>, none, none) -> (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>)
 "func.return"(%Y, %Y_h) : (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_zhigh_gru0_dyn
// CHECK-SAME:   ([[X_:%.+]]: tensor<?x?x?xf32>, [[W_:%.+]]: tensor<1x600x?xf32>, [[R_:%.+]]: tensor<1x600x200xf32>, [[B_:%.+]]: tensor<1x1200xf32>) -> (tensor<?x1x?x200xf32>, tensor<1x?x200xf32>) {
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]]:6 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200]} : (tensor<1x1200xf32>) -> (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#0, [[VAR_0_]]#1, [[VAR_0_]]#2) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#3, [[VAR_0_]]#4, [[VAR_0_]]#5) : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<1x600x?xf32>) -> tensor<1x?x600xf32>
// CHECK:           [[VAR_5_:%.+]]:3 = "onnx.SplitV11"([[VAR_4_]]) {axis = 2 : si64} : (tensor<1x?x600xf32>) -> (tensor<1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.StickForGRU"([[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2) : (tensor<1x?x200xf32>, tensor<1x?x200xf32>, tensor<1x?x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<1x600x200xf32>) -> tensor<1x200x600xf32>
// CHECK:           [[VAR_8_:%.+]]:3 = "onnx.SplitV11"([[VAR_7_]]) {axis = 2 : si64} : (tensor<1x200x600xf32>) -> (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>)
// CHECK:           [[VAR_9_:%.+]] = "zhigh.StickForGRU"([[VAR_8_]]#0, [[VAR_8_]]#1, [[VAR_8_]]#2) : (tensor<1x200x200xf32>, tensor<1x200x200xf32>, tensor<1x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[VAR_10_:%.+]] = "zhigh.GRU"([[VAR_3_]], [[VAR_cst_]], [[VAR_6_]], [[VAR_1_]], [[VAR_9_]], [[VAR_2_]]) {direction = "forward", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.Unstick"([[VAR_10_]]) : (tensor<*xf16>) -> tensor<?x1x?x200xf32>
// CHECK:           [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_11_]], [[VAR_12_]], [[VAR_15_]], [[VAR_13_]], [[VAR_14_]]) : (tensor<?x1x?x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x?x200xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.SqueezeV11"([[VAR_16_]]) {axes = [0]} : (tensor<1x1x?x200xf32>) -> tensor<1x?x200xf32>
// CHECK:           return [[VAR_11_]], [[VAR_17_]] : tensor<?x1x?x200xf32>, tensor<1x?x200xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_zhigh_gru0_bidir_dyn(%X: tensor<?x?x?xf32>, %W: tensor<2x600x?xf32>, %R: tensor<2x600x200xf32>, %B: tensor<2x1200xf32>) -> (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h = "onnx.GRU"(%X, %W, %R, %B, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "bidirectional", hidden_size = 200 : si64, linear_before_reset = 1 : si64, onnx_node_name = "gru" } : (tensor<?x?x?xf32>, tensor<2x600x?xf32>, tensor<2x600x200xf32>, tensor<2x1200xf32>, none, none) -> (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>)
 "func.return"(%Y, %Y_h) : (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_zhigh_gru0_bidir_dyn
// CHECK-SAME:   ([[X_:%.+]]: tensor<?x?x?xf32>, [[W_:%.+]]: tensor<2x600x?xf32>, [[R_:%.+]]: tensor<2x600x200xf32>, [[B_:%.+]]: tensor<2x1200xf32>) -> (tensor<?x2x?x200xf32>, tensor<2x?x200xf32>) {
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]]:6 = "onnx.SplitV11"([[B_]]) {axis = 1 : si64, split = [200, 200, 200, 200, 200, 200]} : (tensor<2x1200xf32>) -> (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#0, [[VAR_0_]]#1, [[VAR_0_]]#2) : (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickForGRU"([[VAR_0_]]#3, [[VAR_0_]]#4, [[VAR_0_]]#5) : (tensor<2x200xf32>, tensor<2x200xf32>, tensor<2x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[X_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[W_]]) {perm = [0, 2, 1]} : (tensor<2x600x?xf32>) -> tensor<2x?x600xf32>
// CHECK:           [[VAR_5_:%.+]]:3 = "onnx.SplitV11"([[VAR_4_]]) {axis = 2 : si64} : (tensor<2x?x600xf32>) -> (tensor<2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.StickForGRU"([[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2) : (tensor<2x?x200xf32>, tensor<2x?x200xf32>, tensor<2x?x200xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[R_]]) {perm = [0, 2, 1]} : (tensor<2x600x200xf32>) -> tensor<2x200x600xf32>
// CHECK:           [[VAR_8_:%.+]]:3 = "onnx.SplitV11"([[VAR_7_]]) {axis = 2 : si64} : (tensor<2x200x600xf32>) -> (tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>)
// CHECK:           [[VAR_9_:%.+]] = "zhigh.StickForGRU"([[VAR_8_]]#0, [[VAR_8_]]#1, [[VAR_8_]]#2) : (tensor<2x200x200xf32>, tensor<2x200x200xf32>, tensor<2x200x200xf32>) -> tensor<*xf16>
// CHECK:           [[VAR_10_:%.+]] = "zhigh.GRU"([[VAR_3_]], [[VAR_cst_]], [[VAR_6_]], [[VAR_1_]], [[VAR_9_]], [[VAR_2_]]) {direction = "bidirectional", hidden_size = 200 : si64, return_all_steps = -1 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.Unstick"([[VAR_10_]]) : (tensor<*xf16>) -> tensor<?x2x?x200xf32>
// CHECK:           [[VAR_16_:%.+]]:2 = "onnx.SplitV11"([[VAR_11_]]) {axis = 1 : si64} : (tensor<?x2x?x200xf32>) -> (tensor<?x1x?x200xf32>, tensor<?x1x?x200xf32>)
// CHECK:           [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_16_]]#0, [[VAR_12_]], [[VAR_15_]], [[VAR_13_]], [[VAR_14_]]) : (tensor<?x1x?x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x?x200xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_16_]]#1, [[VAR_13_]], [[VAR_14_]], [[VAR_13_]], [[VAR_14_]]) : (tensor<?x1x?x200xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x?x200xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Concat"([[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64} : (tensor<1x1x?x200xf32>, tensor<1x1x?x200xf32>) -> tensor<1x2x?x200xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.SqueezeV11"([[VAR_19_]]) {axes = [0]} : (tensor<1x2x?x200xf32>) -> tensor<2x?x200xf32>
// CHECK:           return [[VAR_11_]], [[VAR_20_]] : tensor<?x2x?x200xf32>, tensor<2x?x200xf32>
// CHECK:         }
}

// -----

func.func @gru_with_len(%arg0: tensor<2x2x1xf32>, %arg1: tensor<1x3x1xf32>, %arg2 : tensor<1x3x1xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %lens = onnx.Constant dense<[2, 1]> : tensor<2xi32>
  %cst = "onnx.NoValue"() {value} : () -> none
  %res:2 = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %lens, %cst) {layout = 0 : si64, linear_before_reset = 1 : si64}
    : ( tensor<2x2x1xf32>, tensor<1x3x1xf32>, tensor<1x3x1xf32>, none, tensor<2xi32>, none) -> (tensor<*xf32>, tensor<*xf32>)
 onnx.Return %res#0,  %res#1 : tensor<*xf32>, tensor<*xf32>

// CHECK-LABEL:  func.func @gru_with_len
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x2x1xf32>, [[PARAM_1_:%.+]]: tensor<1x3x1xf32>, [[PARAM_2_:%.+]]: tensor<1x3x1xf32>) -> (tensor<2x1x2x1xf32>, tensor<1x2x1xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[2, 1]> : tensor<2xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<2x2x1xf32>) -> tensor<2x2x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 1]} : (tensor<1x3x1xf32>) -> tensor<1x1x3xf32>
// CHECK:           [[VAR_4_:%.+]]:3 = "onnx.SplitV11"([[VAR_3_]]) {axis = 2 : si64} : (tensor<1x1x3xf32>) -> (tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>)
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.StickForGRU"([[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[PARAM_2_]]) {perm = [0, 2, 1]} : (tensor<1x3x1xf32>) -> tensor<1x1x3xf32>
// CHECK:           [[VAR_7_:%.+]]:3 = "onnx.SplitV11"([[VAR_6_]]) {axis = 2 : si64} : (tensor<1x1x3xf32>) -> (tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>)
// CHECK:           [[VAR_8_:%.+]] = "zhigh.StickForGRU"([[VAR_7_]]#0, [[VAR_7_]]#1, [[VAR_7_]]#2) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<*xf16>
// CHECK:           [[VAR_9_:%.+]] = "zhigh.GRU"([[VAR_2_]], [[VAR_1_]], [[VAR_5_]], [[VAR_1_]], [[VAR_8_]], [[VAR_1_]]) {direction = "forward", hidden_size = 1 : si64, return_all_steps = -1 : si64} : (tensor<2x2x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, tensor<*xf16>, none, tensor<*xf16>, none) -> tensor<*xf16>
// CHECK:           [[VAR_10_:%.+]] = "zhigh.Unstick"([[VAR_9_]]) : (tensor<*xf16>) -> tensor<2x1x2x1xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "zhigh.FixGRUY"([[VAR_10_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<2x1x2x1xf32>, tensor<2xi32>, none) -> tensor<2x1x2x1xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "zhigh.FixGRUYh"([[VAR_10_]], [[VAR_0_]]) : (tensor<2x1x2x1xf32>, tensor<2xi32>) -> tensor<1x2x1xf32>
// CHECK:           onnx.Return [[VAR_11_]], [[VAR_12_]] : tensor<2x1x2x1xf32>, tensor<1x2x1xf32>
// CHECK:         }
}

// -----

// COM : Maximum hidden_size in GRU is 10880. Not lowered when using 10881.

func.func @test_onnx_to_zhigh_gru_exceed_num_hidden(%X: tensor<7x2000x204xf32>, %W: tensor<1x16384x204xf32>, %R: tensor<1x16384x10881xf32>, %B: tensor<1x16386xf32>) -> (tensor<7x1x2000x10881xf32>, tensor<1x2000x10881xf32>) {
 %cst = "onnx.NoValue"() {value} : () -> none
 %Y, %Y_h = "onnx.GRU"(%X, %W, %R, %B, %cst, %cst) { activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 10881 : si64, linear_before_reset = 1 : si64, onnx_node_name = "gru" } : (tensor<7x2000x204xf32>, tensor<1x16384x204xf32>, tensor<1x16384x10881xf32>, tensor<1x16386xf32>, none, none) -> (tensor<7x1x2000x10881xf32>, tensor<1x2000x10881xf32>)
 "func.return"(%Y, %Y_h) : (tensor<7x1x2000x10881xf32>, tensor<1x2000x10881xf32>) -> ()

  // CHECK-LABEL: test_onnx_to_zhigh_gru_exceed_num_hidden
  // CHECK: "onnx.GRU"

}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func.func @test_exceed_limit_gemm(%arg0 : tensor<32769x5xf32>, %arg1 : tensor<5x32769xf32>, %arg2: tensor<32769xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<32769x5xf32>, tensor<5x32769xf32>, tensor<32769xf32>) -> tensor<*xf32>
 "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_gemm
// CHECK:        "onnx.Gemm"

}
