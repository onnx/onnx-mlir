// RUN: onnx-mlir-opt --onnx-dim-analysis %s -split-input-file | FileCheck %s

// -----

// This test is an excerpt of BertSquad-12 model in the model zoo.
// It was normalized via calling `--simplify-shape-related-ops-onnx`
// Expected results: All unknown dimensions have the same group ID that is 0.

func.func @test_dim_analysis_with_bert(%arg0: tensor<?x256xi64>, %arg1: tensor<?x256xi64>) -> (tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>) {
  %0 = "onnx.Dim"(%arg1) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  %1 = onnx.Constant dense<256> : tensor<1xi64>
  %2 = onnx.Constant dense<1> : tensor<1xi64>
  %3 = "onnx.Concat"(%0, %1, %2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %4 = "onnx.ConstantOfShape"(%3) {onnx_node_name = "bert/encoder/ones", value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<?x256x1xf32>
  %5 = onnx.Constant dense<1> : tensor<1xi64>
  %6 = onnx.Constant dense<256> : tensor<1xi64>
  %7 = "onnx.Concat"(%0, %5, %6) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %8 = "onnx.Reshape"(%arg0, %7) {allowzero = 0 : si64, onnx_node_name = "bert/encoder/Reshape"} : (tensor<?x256xi64>, tensor<3xi64>) -> tensor<?x1x256xi64>
  %9 = "onnx.Cast"(%8) {onnx_node_name = "bert/encoder/Cast", to = f32} : (tensor<?x1x256xi64>) -> tensor<?x1x256xf32>
  %10 = "onnx.Mul"(%4, %9) {onnx_node_name = "bert/encoder/mul"} : (tensor<?x256x1xf32>, tensor<?x1x256xf32>) -> tensor<?x256x256xf32>
  %11 = onnx.Constant dense<[-1, 1, 256, 256]> : tensor<4xi64>
  %12 = "onnx.Reshape"(%10, %11) {allowzero = 0 : si64, onnx_node_name = "bert/encoder/layer_1/attention/self/ExpandDims"} : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x1x256x256xf32>
  %13 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %14 = "onnx.Sub"(%13, %12) {onnx_node_name = "bert/encoder/layer_1/attention/self/sub"} : (tensor<f32>, tensor<?x1x256x256xf32>) -> tensor<?x1x256x256xf32>
  %15 = onnx.Constant dense<-1.000000e+04> : tensor<f32>
  %16 = "onnx.Mul"(%14, %15) {onnx_node_name = "bert/encoder/layer_1/attention/self/mul_1"} : (tensor<?x1x256x256xf32>, tensor<f32>) -> tensor<?x1x256x256xf32>
  %17 = onnx.Constant dense<[-1, 1, 256, 256]> : tensor<4xi64>
  %18 = "onnx.Reshape"(%10, %17) {allowzero = 0 : si64, onnx_node_name = "bert/encoder/layer_0/attention/self/ExpandDims"} : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x1x256x256xf32>
  %19 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %20 = "onnx.Sub"(%19, %18) {onnx_node_name = "bert/encoder/layer_0/attention/self/sub"} : (tensor<f32>, tensor<?x1x256x256xf32>) -> tensor<?x1x256x256xf32>
  %21 = onnx.Constant dense<-1.000000e+04> : tensor<f32>
  %22 = "onnx.Mul"(%20, %21) {onnx_node_name = "bert/encoder/layer_0/attention/self/mul_1"} : (tensor<?x1x256x256xf32>, tensor<f32>) -> tensor<?x1x256x256xf32>
  return %22, %20, %16 : tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>

// CHECK-LABEL:  func.func @test_dim_analysis_with_bert
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x256xi64>, [[PARAM_1_:%.+]]: tensor<?x256xi64>) -> (tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>) {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()

// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_1_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<256> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_4_:%.+]] = onnx.ConstantOfShape([[VAR_3_]]) {onnx_node_name = "bert/encoder/ones", value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<?x256x1xf32>
// CHECK:           "onnx.DimGroup"([[VAR_4_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256x1xf32>) -> ()

// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<256> : tensor<1xi64>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_5_]], [[VAR_6_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_7_]]) {allowzero = 0 : si64, onnx_node_name = "bert/encoder/Reshape"} : (tensor<?x256xi64>, tensor<3xi64>) -> tensor<?x1x256xi64>
// CHECK:           "onnx.DimGroup"([[VAR_8_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256xi64>) -> ()

// CHECK:           [[VAR_9_:%.+]] = "onnx.Cast"([[VAR_8_]]) {onnx_node_name = "bert/encoder/Cast", to = f32} : (tensor<?x1x256xi64>) -> tensor<?x1x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_9_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256xf32>) -> ()

// CHECK:           [[VAR_10_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_9_]]) {onnx_node_name = "bert/encoder/mul"} : (tensor<?x256x1xf32>, tensor<?x1x256xf32>) -> tensor<?x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_10_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256x256xf32>) -> ()

// CHECK:           [[VAR_11_:%.+]] = onnx.Constant dense<[-1, 1, 256, 256]> : tensor<4xi64>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Reshape"([[VAR_10_]], [[VAR_11_]]) {allowzero = 0 : si64, onnx_node_name = "bert/encoder/layer_1/attention/self/ExpandDims"} : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_12_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           [[VAR_13_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Sub"([[VAR_13_]], [[VAR_12_]]) {onnx_node_name = "bert/encoder/layer_1/attention/self/sub"} : (tensor<f32>, tensor<?x1x256x256xf32>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_14_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           [[VAR_15_:%.+]] = onnx.Constant dense<-1.000000e+04> : tensor<f32>
// CHECK:           [[VAR_16_:%.+]] = "onnx.Mul"([[VAR_14_]], [[VAR_15_]]) {onnx_node_name = "bert/encoder/layer_1/attention/self/mul_1"} : (tensor<?x1x256x256xf32>, tensor<f32>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_16_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           [[VAR_17_:%.+]] = onnx.Constant dense<[-1, 1, 256, 256]> : tensor<4xi64>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Reshape"([[VAR_10_]], [[VAR_17_]]) {allowzero = 0 : si64, onnx_node_name = "bert/encoder/layer_0/attention/self/ExpandDims"} : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_18_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           [[VAR_19_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Sub"([[VAR_19_]], [[VAR_18_]]) {onnx_node_name = "bert/encoder/layer_0/attention/self/sub"} : (tensor<f32>, tensor<?x1x256x256xf32>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_20_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           [[VAR_21_:%.+]] = onnx.Constant dense<-1.000000e+04> : tensor<f32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Mul"([[VAR_20_]], [[VAR_21_]]) {onnx_node_name = "bert/encoder/layer_0/attention/self/mul_1"} : (tensor<?x1x256x256xf32>, tensor<f32>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_22_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           return [[VAR_22_]], [[VAR_20_]], [[VAR_16_]] : tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>
// CHECK:         }
}

// -----

func.func @test_unary_elementwise(%arg0 : tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  "func.return"(%0) : (tensor<?x3x?xf32>) -> ()

// CHECK-LABEL:  func.func @test_unary_elementwise
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 2 : si64, group_id = 1 : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x3x?xf32>) -> ()

// CHECK:           [[VAR_0_:%.+]] = "onnx.Sigmoid"([[PARAM_0_]]) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 2 : si64, group_id = 1 : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x3x?xf32>) -> ()

// CHECK:           return [[VAR_0_]] : tensor<?x3x?xf32>
// CHECK:         }
}

// -----

func.func @test_binary_elementwise(%arg0 : tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  %1 = "onnx.Add"(%0, %arg0) : (tensor<?x3x?xf32>, tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  "func.return"(%1) : (tensor<?x3x?xf32>) -> ()

// CHECK-LABEL:  func.func @test_binary_elementwise
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 2 : si64, group_id = 1 : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x3x?xf32>) -> ()

// CHECK:           [[VAR_0_:%.+]] = "onnx.Sigmoid"([[PARAM_0_]]) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 2 : si64, group_id = 1 : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x3x?xf32>) -> ()

// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_0_]]) : (tensor<?x3x?xf32>, tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 2 : si64, group_id = 1 : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           return [[VAR_1_]] : tensor<?x3x?xf32>
// CHECK:         }
}

// -----

func.func @test_expand_from_concat_dims(%arg0: tensor<1x256xi64>, %arg1: tensor<?x256xi64>) -> tensor<?x256xi64> {
  %0 = onnx.Constant dense<256> : tensor<1xi64>
  %1 = "onnx.Dim"(%arg1) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  %2 = "onnx.Concat"(%1, %0) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %3 = "onnx.Expand"(%arg0, %2) {onnx_node_name = "Expand_30"} : (tensor<1x256xi64>, tensor<2xi64>) -> tensor<?x256xi64>
  return %3: tensor<?x256xi64>

// CHECK-LABEL:  func.func @test_expand_from_concat_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x256xi64>, [[PARAM_1_:%.+]]: tensor<?x256xi64>) -> tensor<?x256xi64> {
// CHECK:           "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<256> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_1_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK:           "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_0_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Expand"([[PARAM_0_]], [[VAR_2_]]) {onnx_node_name = "Expand_30"} : (tensor<1x256xi64>, tensor<2xi64>) -> tensor<?x256xi64>
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()
// CHECK:           return [[VAR_3_]] : tensor<?x256xi64>
// CHECK:         }
}
