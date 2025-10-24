// RUN: onnx-mlir-opt --recompose-onnx="recompose-layernorm-by-transpose" --canonicalize %s -split-input-file | FileCheck %s

func.func @main_graph(%arg0: tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32> {
  %2 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %4 = onnx.Constant dense<9.99999997E-7> : tensor<f32>
  %5 = onnx.Constant dense<[[[0.0970484465]], [[0.0882187337]], [[0.135120019]], [[0.14907673]]]> : tensor<4x1x1xf32>
  %6 = onnx.Constant dense<[[[0.191972837]], [[0.286264896]], [[0.180535644]], [[0.166878015]]]> : tensor<4x1x1xf32>    
  %9 = "onnx.ReduceMeanV13"(%arg0) {axes = [1], keepdims = 1 : si64, onnx_node_name = "/mask_downscaling/mask_downscaling.1/ReduceMean"} : (tensor<1x4x128x128xf32>) -> tensor<1x1x128x128xf32>
  %10 = "onnx.Sub"(%arg0, %9) {onnx_node_name = "/mask_downscaling/mask_downscaling.1/Sub"} : (tensor<1x4x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x4x128x128xf32>
  %11 = "onnx.Mul"(%10, %10) {onnx_node_name = "/mask_downscaling/mask_downscaling.1/Pow_1"} : (tensor<1x4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
  %12 = "onnx.ReduceMeanV13"(%11) {axes = [1], keepdims = 1 : si64, onnx_node_name = "/mask_downscaling/mask_downscaling.1/ReduceMean_1"} : (tensor<1x4x128x128xf32>) -> tensor<1x1x128x128xf32>
  %13 = "onnx.Add"(%12, %4) {onnx_node_name = "/mask_downscaling/mask_downscaling.1/Add"} : (tensor<1x1x128x128xf32>, tensor<f32>) -> tensor<1x1x128x128xf32>
  %14 = "onnx.Sqrt"(%13) {onnx_node_name = "/mask_downscaling/mask_downscaling.1/Sqrt"} : (tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32>
  %15 = "onnx.Div"(%10, %14) {onnx_node_name = "/mask_downscaling/mask_downscaling.1/Div"} : (tensor<1x4x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x4x128x128xf32>
  %16 = "onnx.Mul"(%15, %5) {onnx_node_name = "/mask_downscaling/mask_downscaling.1/Mul_2"} : (tensor<1x4x128x128xf32>, tensor<4x1x1xf32>) -> tensor<1x4x128x128xf32>
  %17 = "onnx.Add"(%16, %6) {onnx_node_name = "/mask_downscaling/mask_downscaling.1/Add_1"} : (tensor<1x4x128x128xf32>, tensor<4x1x1xf32>) -> tensor<1x4x128x128xf32>
  return %17 : tensor<1x4x128x128xf32>
}
// CHECK-LABEL:  func.func @main_graph
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}{{.}}[0.0970484465]{{.}}, {{.}}[0.0882187337]{{.}}, {{.}}[0.135120019]{{.}}, {{.}}[0.14907673]{{.}}{{.}}> : tensor<4x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<{{.}}{{.}}[0.191972837]{{.}}, {{.}}[0.286264896]{{.}}, {{.}}[0.180535644]{{.}}, {{.}}[0.166878015]{{.}}{{.}}> : tensor<4x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 4, 1, 1]> : tensor<4xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_0_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<4x1x1xf32>, tensor<4xi64>) -> tensor<1x4x1x1xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [0, 2, 3, 1]} : (tensor<1x4x1x1xf32>) -> tensor<1x1x1x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 2, 3, 1]} : (tensor<1x4x128x128xf32>) -> tensor<1x128x128x4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Reshape"([[VAR_1_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<4x1x1xf32>, tensor<4xi64>) -> tensor<1x4x1x1xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [0, 2, 3, 1]} : (tensor<1x4x1x1xf32>) -> tensor<1x1x1x4xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_5_]], [[VAR_4_]], [[VAR_7_]]) {axis = 3 : si64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : si64} : (tensor<1x128x128x4xf32>, tensor<1x1x1x4xf32>, tensor<1x1x1x4xf32>) -> (tensor<1x128x128x4xf32>, none, none)
// CHECK:           [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_Y_]]) {perm = [0, 3, 1, 2]} : (tensor<1x128x128x4xf32>) -> tensor<1x4x128x128xf32>
// CHECK:           return [[VAR_8_]] : tensor<1x4x128x128xf32>
// CHECK:         }

// -----

// TODO: ADD more lit tests here
