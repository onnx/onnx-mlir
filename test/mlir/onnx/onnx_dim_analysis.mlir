// RUN: onnx-mlir-opt --onnx-dim-analysis %s -split-input-file | FileCheck %s

// Check if dim_analysis takes into account the relationship between inputs via dim_params.
func.func @test_dim_params_onnx_return(%arg0: tensor<?x?xf32> {onnx.dim_params = "0:M,1:N", onnx.name = "X"}, %arg1: tensor<?x?xf32> {onnx.dim_params = "0:M,1:P", onnx.name = "Y"}) -> (tensor<?x?xf32> {onnx.dim_params = "0:M,1:N", onnx.name = "Z"}) {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %0: tensor<?x?xf32>

// CHECK-LABEL:  func.func @test_dim_params_onnx_return
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32> {onnx.dim_params = "0:M,1:N", onnx.name = "X"}, [[PARAM_1_:%.+]]: tensor<?x?xf32> {onnx.dim_params = "0:M,1:P", onnx.name = "Y"}) -> (tensor<?x?xf32> {onnx.dim_params = "0:M,1:N", onnx.name = "Z"}) {
// CHECK-DAG:           "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 1 : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:           "onnx.DimGroup"([[PARAM_1_]]) {axis = 1 : si64, group_id = 4 : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:           "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:           "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x?xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-DAG:           "onnx.DimGroup"([[VAR_0_]]) {axis = 1 : si64, group_id = 1 : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:           "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x?xf32>) -> ()
// CHECK:           onnx.Return [[VAR_0_]] : tensor<?x?xf32>
// CHECK:         }
}

// -----

// Check if dim_analysis takes into account the relationship between inputs via dim_params.
func.func @test_dim_params_std_return(%arg0: tensor<?x?xf32> {onnx.dim_params = "0:M,1:N", onnx.name = "X"}, %arg1: tensor<?x?xf32> {onnx.dim_params = "0:M,1:P", onnx.name = "Y"}) -> (tensor<?x?xf32> {onnx.dim_params = "0:M,1:N", onnx.name = "Z"}) {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0: tensor<?x?xf32>

// CHECK-LABEL:  func.func @test_dim_params_std_return
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32> {onnx.dim_params = "0:M,1:N", onnx.name = "X"}, [[PARAM_1_:%.+]]: tensor<?x?xf32> {onnx.dim_params = "0:M,1:P", onnx.name = "Y"}) -> (tensor<?x?xf32> {onnx.dim_params = "0:M,1:N", onnx.name = "Z"}) {
// CHECK-DAG:           "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 1 : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:           "onnx.DimGroup"([[PARAM_1_]]) {axis = 1 : si64, group_id = 4 : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:           "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:           "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x?xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-DAG:           "onnx.DimGroup"([[VAR_0_]]) {axis = 1 : si64, group_id = 1 : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:           "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x?xf32>) -> ()
// CHECK:           return [[VAR_0_]] : tensor<?x?xf32>
// CHECK:         }
}

// -----

// This test is an excerpt of BertSquad-12 model in the model zoo.
// It was normalized via calling `--simplify-shape-related-ops-onnx`
// Expected results: All unknown dimensions have the same group ID that is 0.

func.func @test_dim_analysis_with_bert(%arg0: tensor<?x256xi64>, %arg1: tensor<?x256xi64>) -> (tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>) {
  %0 = "onnx.Dim"(%arg1) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  %1 = onnx.Constant dense<256> : tensor<1xi64>
  %2 = onnx.Constant dense<1> : tensor<1xi64>
  %3 = "onnx.Concat"(%0, %1, %2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %4 = "onnx.ConstantOfShape"(%3) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<?x256x1xf32>
  %5 = onnx.Constant dense<1> : tensor<1xi64>
  %6 = onnx.Constant dense<256> : tensor<1xi64>
  %7 = "onnx.Concat"(%0, %5, %6) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %8 = "onnx.Reshape"(%arg0, %7) {allowzero = 0 : si64} : (tensor<?x256xi64>, tensor<3xi64>) -> tensor<?x1x256xi64>
  %9 = "onnx.Cast"(%8) {to = f32} : (tensor<?x1x256xi64>) -> tensor<?x1x256xf32>
  %10 = "onnx.Mul"(%4, %9) : (tensor<?x256x1xf32>, tensor<?x1x256xf32>) -> tensor<?x256x256xf32>
  %11 = onnx.Constant dense<[-1, 1, 256, 256]> : tensor<4xi64>
  %12 = "onnx.Reshape"(%10, %11) {allowzero = 0 : si64} : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x1x256x256xf32>
  %13 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %14 = "onnx.Sub"(%13, %12) : (tensor<f32>, tensor<?x1x256x256xf32>) -> tensor<?x1x256x256xf32>
  %15 = onnx.Constant dense<-1.000000e+04> : tensor<f32>
  %16 = "onnx.Mul"(%14, %15) : (tensor<?x1x256x256xf32>, tensor<f32>) -> tensor<?x1x256x256xf32>
  %17 = onnx.Constant dense<[-1, 1, 256, 256]> : tensor<4xi64>
  %18 = "onnx.Reshape"(%10, %17) {allowzero = 0 : si64} : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x1x256x256xf32>
  %19 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %20 = "onnx.Sub"(%19, %18) : (tensor<f32>, tensor<?x1x256x256xf32>) -> tensor<?x1x256x256xf32>
  %21 = onnx.Constant dense<-1.000000e+04> : tensor<f32>
  %22 = "onnx.Mul"(%20, %21) : (tensor<?x1x256x256xf32>, tensor<f32>) -> tensor<?x1x256x256xf32>
  onnx.Return %22, %20, %16 : tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>

// CHECK-LABEL:  func.func @test_dim_analysis_with_bert
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x256xi64>, [[PARAM_1_:%.+]]: tensor<?x256xi64>) -> (tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>) {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()

// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_1_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<256> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_4_:%.+]] = onnx.ConstantOfShape([[VAR_3_]]) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<?x256x1xf32>
// CHECK:           "onnx.DimGroup"([[VAR_4_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256x1xf32>) -> ()

// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<256> : tensor<1xi64>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_5_]], [[VAR_6_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_7_]]) {allowzero = 0 : si64} : (tensor<?x256xi64>, tensor<3xi64>) -> tensor<?x1x256xi64>
// CHECK:           "onnx.DimGroup"([[VAR_8_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256xi64>) -> ()

// CHECK:           [[VAR_9_:%.+]] = "onnx.Cast"([[VAR_8_]]) {saturate = 1 : si64, to = f32} : (tensor<?x1x256xi64>) -> tensor<?x1x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_9_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256xf32>) -> ()

// CHECK:           [[VAR_10_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_9_]]) : (tensor<?x256x1xf32>, tensor<?x1x256xf32>) -> tensor<?x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_10_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256x256xf32>) -> ()

// CHECK:           [[VAR_11_:%.+]] = onnx.Constant dense<[-1, 1, 256, 256]> : tensor<4xi64>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Reshape"([[VAR_10_]], [[VAR_11_]]) {allowzero = 0 : si64} : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_12_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           [[VAR_13_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Sub"([[VAR_13_]], [[VAR_12_]]) : (tensor<f32>, tensor<?x1x256x256xf32>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_14_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           [[VAR_15_:%.+]] = onnx.Constant dense<-1.000000e+04> : tensor<f32>
// CHECK:           [[VAR_16_:%.+]] = "onnx.Mul"([[VAR_14_]], [[VAR_15_]]) : (tensor<?x1x256x256xf32>, tensor<f32>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_16_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           [[VAR_17_:%.+]] = onnx.Constant dense<[-1, 1, 256, 256]> : tensor<4xi64>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Reshape"([[VAR_10_]], [[VAR_17_]]) {allowzero = 0 : si64} : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_18_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           [[VAR_19_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Sub"([[VAR_19_]], [[VAR_18_]]) : (tensor<f32>, tensor<?x1x256x256xf32>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_20_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           [[VAR_21_:%.+]] = onnx.Constant dense<-1.000000e+04> : tensor<f32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Mul"([[VAR_20_]], [[VAR_21_]]) : (tensor<?x1x256x256xf32>, tensor<f32>) -> tensor<?x1x256x256xf32>
// CHECK:           "onnx.DimGroup"([[VAR_22_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x1x256x256xf32>) -> ()

// CHECK:           onnx.Return [[VAR_22_]], [[VAR_20_]], [[VAR_16_]] : tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>
// CHECK:         }
}

// -----

func.func @test_unary_elementwise(%arg0 : tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  "onnx.Return"(%0) : (tensor<?x3x?xf32>) -> ()

// CHECK-LABEL:  func.func @test_unary_elementwise
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 2 : si64, group_id = [[GROUP_0_:.*]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = [[GROUP_1_:.*]]  : si64} : (tensor<?x3x?xf32>) -> ()

// CHECK:           [[VAR_0_:%.+]] = "onnx.Sigmoid"([[PARAM_0_]]) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 2 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x3x?xf32>) -> ()

// CHECK:           onnx.Return [[VAR_0_]] : tensor<?x3x?xf32>
// CHECK:         }
}

// -----

func.func @test_binary_elementwise(%arg0 : tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  %1 = "onnx.Add"(%0, %arg0) : (tensor<?x3x?xf32>, tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  "onnx.Return"(%1) : (tensor<?x3x?xf32>) -> ()

// CHECK-LABEL:  func.func @test_binary_elementwise
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 2 : si64, group_id = [[GROUP_0_:.*]]  : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = [[GROUP_1_:.*]]  : si64} : (tensor<?x3x?xf32>) -> ()

// CHECK:           [[VAR_0_:%.+]] = "onnx.Sigmoid"([[PARAM_0_]]) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 2 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x3x?xf32>) -> ()

// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_0_]]) : (tensor<?x3x?xf32>, tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 2 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 0 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           onnx.Return [[VAR_1_]] : tensor<?x3x?xf32>
// CHECK:         }
}

// -----

func.func @test_matmul_batchsize(%arg0: tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32>
  %1 = "onnx.MatMul"(%0, %arg0) : (tensor<?x8x16x16xf32>, tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32>
  "onnx.Return"(%1) : (tensor<?x8x16x16xf32>) -> ()

// CHECK-LABEL:  func.func @test_matmul_batchsize
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32> {
// CHECK:           "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x8x16x16xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.Sigmoid"([[PARAM_0_]]) : (tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32>
// CHECK:           "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x8x16x16xf32>) -> ()
// CHECK:           [[VAR_1_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[PARAM_0_]]) : (tensor<?x8x16x16xf32>, tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32>
// CHECK:           "onnx.DimGroup"([[VAR_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x8x16x16xf32>) -> ()
// CHECK:           onnx.Return [[VAR_1_]] : tensor<?x8x16x16xf32>
// CHECK:         }
}

// -----

func.func @test_matmul_batchsize_diff_rank(%arg0: tensor<8x?x16x4xf32>) -> tensor<8x?x16x32xf32> {
  %shape = onnx.Constant dense<[-1, 4, 128]> : tensor<3xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} : (tensor<8x?x16x4xf32>, tensor<3xi64>) -> tensor<?x4x32xf32>
  %1 = "onnx.MatMul"(%arg0, %0) : (tensor<8x?x16x4xf32>, tensor<?x4x32xf32>) -> tensor<8x?x16x32xf32>
  "onnx.Return"(%1) : (tensor<8x?x16x32xf32>) -> ()

// CHECK-LABEL:  func.func @test_matmul_batchsize_diff_rank
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8x?x16x4xf32>) -> tensor<8x?x16x32xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 0 : si64} : (tensor<8x?x16x4xf32>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[-1, 4, 128]> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<8x?x16x4xf32>, tensor<3xi64>) -> tensor<?x4x32xf32>
// CHECK:           "onnx.DimGroup"([[VAR_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x4x32xf32>) -> ()
// CHECK:           [[VAR_2_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[VAR_1_]]) : (tensor<8x?x16x4xf32>, tensor<?x4x32xf32>) -> tensor<8x?x16x32xf32>
// CHECK:           "onnx.DimGroup"([[VAR_2_]]) {axis = 1 : si64, group_id = 0 : si64} : (tensor<8x?x16x32xf32>) -> ()
// CHECK:           onnx.Return [[VAR_2_]] : tensor<8x?x16x32xf32>
// CHECK:         }
}

// -----

func.func @test_reshape_single_dyn_dim(%arg0: tensor<8x?x16x4xf32>) -> tensor<?x4x32xf32> {
  %shape = onnx.Constant dense<[-1, 4, 128]> : tensor<3xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} : (tensor<8x?x16x4xf32>, tensor<3xi64>) -> tensor<?x4x32xf32>
  "onnx.Return"(%0) : (tensor<?x4x32xf32>) -> ()

// CHECK-LABEL:  func.func @test_reshape_single_dyn_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8x?x16x4xf32>) -> tensor<?x4x32xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 0 : si64} : (tensor<8x?x16x4xf32>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[-1, 4, 128]> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<8x?x16x4xf32>, tensor<3xi64>) -> tensor<?x4x32xf32>
// CHECK:           "onnx.DimGroup"([[VAR_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x4x32xf32>) -> ()
// CHECK:           onnx.Return [[VAR_1_]] : tensor<?x4x32xf32>
// CHECK:         }
}

// -----

func.func @test_reshape_allowzero(%arg0: tensor<?x?x768xf32>) -> tensor<?x?x12x64xf32> {
  %184 = onnx.Constant dense<[0, 0, 12, 64]> : tensor<4xi64>
  %494 = "onnx.Reshape"(%arg0, %184) {allowzero = 0 : si64} : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x12x64xf32>
  "onnx.Return"(%494) : (tensor<?x?x12x64xf32>) -> ()

// CHECK-LABEL:  func.func @test_reshape_allowzero
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xf32>) -> tensor<?x?x12x64xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = [[GROUP_0_:.*]] : si64} : (tensor<?x?x768xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = [[GROUP_1_:.*]] : si64} : (tensor<?x?x768xf32>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0, 0, 12, 64]> : tensor<4xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x12x64xf32>
// CHECK:           "onnx.DimGroup"([[VAR_1_]]) {axis = 1 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x12x64xf32>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_1_]]) {axis = 0 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x12x64xf32>) -> ()
// CHECK:           onnx.Return [[VAR_1_]] : tensor<?x?x12x64xf32>
// CHECK:         }
}

// -----

func.func @test_expand_from_concat_dims(%arg0: tensor<1x256xi64>, %arg1: tensor<?x256xi64>) -> tensor<?x256xi64> {
  %0 = onnx.Constant dense<256> : tensor<1xi64>
  %1 = "onnx.Dim"(%arg1) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  %2 = "onnx.Concat"(%1, %0) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %3 = "onnx.Expand"(%arg0, %2) {onnx_node_name = "Expand_30"} : (tensor<1x256xi64>, tensor<2xi64>) -> tensor<?x256xi64>
  onnx.Return %3: tensor<?x256xi64>

// CHECK-LABEL:  func.func @test_expand_from_concat_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x256xi64>, [[PARAM_1_:%.+]]: tensor<?x256xi64>) -> tensor<?x256xi64> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<256> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_1_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_0_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Expand"([[PARAM_0_]], [[VAR_2_]]) {onnx_node_name = "Expand_30"} : (tensor<1x256xi64>, tensor<2xi64>) -> tensor<?x256xi64>
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()
// CHECK:           onnx.Return [[VAR_3_]] : tensor<?x256xi64>
// CHECK:         }
}

// -----

// COM: input and output have the same rank of 2, and if one output dim is
// from an input dim, the other output dim must be from the remaining input dim.

func.func @test_reshape_rank_2(%arg0: tensor<?x?xi64>) -> tensor<?x?xi64> {
  %cst_minus1 = onnx.Constant dense<-1> : tensor<1xi64>
  %0 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<?x?xi64>) -> tensor<1xi64>
  %1 = "onnx.Concat"(%0, %cst_minus1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %2 = "onnx.Reshape"(%arg0, %1) {allowzero = 0 : si64} : (tensor<?x?xi64>, tensor<2xi64>) -> tensor<?x?xi64>
  onnx.Return %2: tensor<?x?xi64>

// CHECK-LABEL:  func.func @test_reshape_rank_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xi64>) -> tensor<?x?xi64> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 0 : si64} : (tensor<?x?xi64>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64} : (tensor<?x?xi64>) -> tensor<1xi64>
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 2 : si64} : (tensor<?x?xi64>) -> ()
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_0_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<?x?xi64>, tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 1 : si64, group_id = 2 : si64} : (tensor<?x?xi64>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x?xi64>) -> ()
// CHECK:           onnx.Return [[VAR_3_]] : tensor<?x?xi64>
// CHECK:         }
}

// -----

func.func @test_expand_from_concat_dims(%arg0: tensor<1x256xi64>, %arg1: tensor<?x256xi64>) -> tensor<?x256xi64> {
  %0 = onnx.Constant dense<256> : tensor<1xi64>
  %1 = "onnx.Dim"(%arg1) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  %2 = "onnx.Concat"(%1, %0) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %3 = "onnx.Expand"(%arg0, %2) : (tensor<1x256xi64>, tensor<2xi64>) -> tensor<?x256xi64>
  return %3: tensor<?x256xi64>

// CHECK-LABEL:  func.func @test_expand_from_concat_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x256xi64>, [[PARAM_1_:%.+]]: tensor<?x256xi64>) -> tensor<?x256xi64> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<256> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_1_]]) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_0_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Expand"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x256xi64>, tensor<2xi64>) -> tensor<?x256xi64>
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x256xi64>) -> ()
// CHECK:           return [[VAR_3_]] : tensor<?x256xi64>
// CHECK:         }
}

// -----

// COM: Dimension in output tensor is dimension in input tensor multiplied by value in repeats tensor.
// (output_dim[i] = input_dim[i] * repeats[i])
// If input_dim[i] is 1, output_dim[i] and repeats[i] are equal.

func.func @test_tile_input_dim_1(%arg0: tensor<?x?xi64>, %arg1: tensor<1x1xi64>) -> tensor<?x?xi64> {
  %0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x?xi64>) -> tensor<1xi64>
  %1 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<?x?xi64>) -> tensor<1xi64>
  %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %3 = "onnx.Tile"(%arg1, %2) : (tensor<1x1xi64>, tensor<2xi64>) -> tensor<?x?xi64>
  onnx.Return %3: tensor<?x?xi64>

// CHECK-LABEL:  func.func @test_tile_input_dim_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xi64>, [[PARAM_1_:%.+]]: tensor<1x1xi64>) -> tensor<?x?xi64> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_:.*]] : si64} : (tensor<?x?xi64>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = [[GROUP_1_:.*]] : si64} : (tensor<?x?xi64>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x?xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64} : (tensor<?x?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Tile"([[PARAM_1_]], [[VAR_2_]]) : (tensor<1x1xi64>, tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 1 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?xi64>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?xi64>) -> ()
// CHECK:           onnx.Return [[VAR_3_]] : tensor<?x?xi64>
// CHECK:         }
}

// -----

func.func @test_center_crop_pad_1(%arg0: tensor<?x?x8xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?x8xf32> {
  %0 = "onnx.Dim"(%arg1) {axis = 0 : si64} : (tensor<?x?xf32>) -> tensor<1xi64>
  %1 = "onnx.Dim"(%arg1) {axis = 1 : si64} : (tensor<?x?xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %2 = "onnx.CenterCropPad"(%arg0, %shape) {axes = [0, -2]} : (tensor<?x?x8xf32>, tensor<2xi64>) -> tensor<?x?x8xf32>
  return %2 : tensor<?x?x8xf32>

// CHECK-LABEL:  func.func @test_center_crop_pad_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x8xf32>, [[PARAM_1_:%.+]]: tensor<?x?xf32>) -> tensor<?x?x8xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = [[GROUP_0_:.*]] : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 1 : si64, group_id = [[GROUP_1_:.*]] : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_1_]]) {axis = 0 : si64} : (tensor<?x?xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_1_]]) {axis = 1 : si64} : (tensor<?x?xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.CenterCropPad"([[PARAM_0_]], [[VAR_2_]]) {axes = [0, -2]} : (tensor<?x?x8xf32>, tensor<2xi64>) -> tensor<?x?x8xf32>
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 1 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x8xf32>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x8xf32>) -> ()
// CHECK:           return [[VAR_3_]] : tensor<?x?x8xf32>
// CHECK:         }
}

// -----

func.func @test_center_crop_pad_2(%arg0: tensor<?x8x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x8x?xf32> {
  %0 = "onnx.Dim"(%arg1) {axis = 0 : si64} : (tensor<?x?xf32>) -> tensor<1xi64>
  %1 = "onnx.Dim"(%arg1) {axis = 1 : si64} : (tensor<?x?xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %2 = "onnx.CenterCropPad"(%arg0, %shape) {axes = [-3, 2]} : (tensor<?x8x?xf32>, tensor<2xi64>) -> tensor<?x8x?xf32>
  return %2 : tensor<?x8x?xf32>

// CHECK-LABEL:  func.func @test_center_crop_pad_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x8x?xf32>, [[PARAM_1_:%.+]]: tensor<?x?xf32>) -> tensor<?x8x?xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 1 : si64, group_id = [[GROUP_0_:.*]] : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = [[GROUP_1_:.*]] : si64} : (tensor<?x?xf32>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_1_]]) {axis = 0 : si64} : (tensor<?x?xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_1_]]) {axis = 1 : si64} : (tensor<?x?xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.CenterCropPad"([[PARAM_0_]], [[VAR_2_]]) {axes = [-3, 2]} : (tensor<?x8x?xf32>, tensor<2xi64>) -> tensor<?x8x?xf32>
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 2 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x8x?xf32>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 0 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x8x?xf32>) -> ()
// CHECK:           return [[VAR_3_]] : tensor<?x8x?xf32>
// CHECK:         }
}

// -----

func.func @test_max_unpool(%arg0: tensor<1x1x2x2xf32>, %arg1: tensor<1x1x2x2xi64>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.Dim"(%arg2) {axis = 0 : si64} : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
  %1 = "onnx.Dim"(%arg2) {axis = 1 : si64} : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
  %2 = "onnx.Dim"(%arg2) {axis = 2 : si64} : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
  %3 = "onnx.Dim"(%arg2) {axis = 3 : si64} : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
  %output_shape = "onnx.Concat"(%0, %1, %2, %3) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %4 = "onnx.MaxUnpool"(%arg0, %arg1, %output_shape) {kernel_shape = [2, 2], strides = [2, 2]} : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xi64>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %4 : tensor<?x?x?x?xf32>

// CHECK-LABEL:  func.func @test_max_unpool
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x2xf32>, [[PARAM_1_:%.+]]: tensor<1x1x2x2xi64>, [[PARAM_2_:%.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_2_]]) {axis = 0 : si64, group_id = [[GROUP_0_:.*]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_2_]]) {axis = 1 : si64, group_id = [[GROUP_1_:.*]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_2_]]) {axis = 2 : si64, group_id = [[GROUP_2_:.*]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_2_]]) {axis = 3 : si64, group_id = [[GROUP_3_:.*]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_2_]]) {axis = 0 : si64} : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_2_]]) {axis = 1 : si64} : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Dim"([[PARAM_2_]]) {axis = 2 : si64} : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Dim"([[PARAM_2_]]) {axis = 3 : si64} : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.MaxUnpool"([[PARAM_0_]], [[PARAM_1_]], [[VAR_4_]]) {kernel_shape = [2, 2], strides = [2, 2]} : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xi64>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_5_]]) {axis = 3 : si64, group_id = [[GROUP_3_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_5_]]) {axis = 1 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_5_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_5_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK:           return [[VAR_5_]] : tensor<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @test_correct_dimgroup_axis_for_onnx_dim(%arg0: tensor<1x?xi64>) -> tensor<1x1x1x?xf32> {
  %0 = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<1x?xi64>) -> tensor<1xi64>
  %2 = "onnx.Concat"(%0, %0, %0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %3 = onnx.ConstantOfShape(%2) {value = dense<0.000000e+00> : tensor<1xf32>} : (tensor<4xi64>) -> tensor<1x1x1x?xf32>
  return %3: tensor<1x1x1x?xf32> 

// CHECK-LABEL:  func.func @test_correct_dimgroup_axis_for_onnx_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?xi64>) -> tensor<1x1x1x?xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 0 : si64} : (tensor<1x?xi64>) -> ()
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64} : (tensor<1x?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_0_]], [[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_3_:%.+]] = onnx.ConstantOfShape([[VAR_2_]]) {value = dense<0.000000e+00> : tensor<1xf32>} : (tensor<4xi64>) -> tensor<1x1x1x?xf32>
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 3 : si64, group_id = 0 : si64} : (tensor<1x1x1x?xf32>) -> ()
// CHECK:           return [[VAR_3_]] : tensor<1x1x1x?xf32>
// CHECK:         }
}  

// -----

func.func @test_matmul_reduction_dimension(%arg0: tensor<5x?xf32>, %arg1: tensor<?x10xf32>) -> tensor<5x10xf32> {
  %0 =  "onnx.MatMul"(%arg0, %arg1) : (tensor<5x?xf32>, tensor<?x10xf32>) -> tensor<5x10xf32> 
  return %0 : tensor<5x10xf32>
// CHECK-LABEL:  func.func @test_matmul_reduction_dimension
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x?xf32>, [[PARAM_1_:%.+]]: tensor<?x10xf32>) -> tensor<5x10xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x10xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 0 : si64} : (tensor<5x?xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<5x?xf32>, tensor<?x10xf32>) -> tensor<5x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<5x10xf32>
// CHECK:         }
}

// -----

func.func @test_gemm_reduction_dimension(%arg0: tensor<5x?xf32>, %arg1: tensor<?x10xf32>, %arg2: tensor<10xf32>) -> tensor<5x10xf32> {
  %0 =  "onnx.Gemm"(%arg0, %arg1, %arg2) : (tensor<5x?xf32>, tensor<?x10xf32>, tensor<10xf32>) -> tensor<5x10xf32> 
  return %0 : tensor<5x10xf32>

// CHECK-LABEL:  func.func @test_gemm_reduction_dimension
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x?xf32>, [[PARAM_1_:%.+]]: tensor<?x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<5x10xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 0 : si64} : (tensor<5x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x10xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gemm"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<5x?xf32>, tensor<?x10xf32>, tensor<10xf32>) -> tensor<5x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<5x10xf32>
// CHECK:         }
}

// -----

func.func @test_gemm_reduction_dimension_trans(%arg0: tensor<?x5xf32>, %arg1: tensor<10x?xf32>, %arg2: tensor<10xf32>) -> tensor<5x10xf32> {
  %0 =  "onnx.Gemm"(%arg0, %arg1, %arg2) {transA = 1 : si64, transB = 1 : si64} : (tensor<?x5xf32>, tensor<10x?xf32>, tensor<10xf32>) -> tensor<5x10xf32> 
  return %0 : tensor<5x10xf32>

// CHECK-LABEL:  func.func @test_gemm_reduction_dimension_trans
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x5xf32>, [[PARAM_1_:%.+]]: tensor<10x?xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<5x10xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x5xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 1 : si64, group_id = 0 : si64} : (tensor<10x?xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gemm"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<?x5xf32>, tensor<10x?xf32>, tensor<10xf32>) -> tensor<5x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<5x10xf32>
// CHECK:         }
}

// -----

func.func @test_concat_input_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>

// CHECK-LABEL:  func.func @test_concat_input_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>, [[PARAM_1_:%.+]]: tensor<?x?x?xf32>, [[PARAM_2_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_:.*]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_2_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = [[GROUP_1_0_:.*]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 1 : si64, group_id = [[GROUP_1_1_:.*]]  : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_2_]]) {axis = 1 : si64, group_id = [[GROUP_1_2_:.*]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 2 : si64, group_id = [[GROUP_2_:.*]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_1_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_2_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 1 : si64} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 1 : si64, group_id = [[GROUP_1_:.*]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32>
// CHECK:         }
}

// -----

func.func @test_lstm_input_dims(%X: tensor<?x?x10xf32>, %W: tensor<1x16x10xf32>, %R: tensor<1x16x4xf32>, %B: tensor<1x32xf32>, %seq_len: tensor<?xi32>, %initial_h: tensor<?x?x4xf32>, %initial_c: tensor<?x?x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %seq_len, %initial_h, %initial_c, %cst) {hidden_size = 4 : si64} : (tensor<?x?x10xf32>, tensor<1x16x10xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, tensor<?xi32>, tensor<?x?x4xf32>, tensor<?x?x4xf32>, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func.func @test_lstm_input_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x10xf32>, [[PARAM_1_:%.+]]: tensor<1x16x10xf32>, [[PARAM_2_:%.+]]: tensor<1x16x4xf32>, [[PARAM_3_:%.+]]: tensor<1x32xf32>, [[PARAM_4_:%.+]]: tensor<?xi32>, [[PARAM_5_:%.+]]: tensor<?x?x4xf32>, [[PARAM_6_:%.+]]: tensor<?x?x4xf32>) -> tensor<*xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_6_]]) {axis = 0 : si64, group_id = 5 : si64} : (tensor<?x?x4xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_5_]]) {axis = 0 : si64, group_id = 3 : si64} : (tensor<?x?x4xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_5_]]) {axis = 1 : si64, group_id = 2 : si64} : (tensor<?x?x4xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_6_]]) {axis = 1 : si64, group_id = 2 : si64} : (tensor<?x?x4xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_4_]]) {axis = 0 : si64, group_id = 2 : si64} : (tensor<?xi32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 2 : si64} : (tensor<?x?x10xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x?x10xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[Y_h_:%.+]], [[VAR_Y_c_:%.+]] = "onnx.LSTM"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[VAR_0_]]) {direction = "forward", hidden_size = 4 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<?x?x10xf32>, tensor<1x16x10xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, tensor<?xi32>, tensor<?x?x4xf32>, tensor<?x?x4xf32>, none) -> (none, tensor<*xf32>, none)
// CHECK:           return [[Y_h_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_gru_input_dims(%X: tensor<?x?x10xf32>, %W: tensor<1x12x10xf32>, %R: tensor<1x12x4xf32>, %B: tensor<1x24xf32>, %seq_len: tensor<?xi32>, %initial_h: tensor<?x?x4xf32>) -> tensor<*xf32> {
  %Y, %Y_h = "onnx.GRU"(%X, %W, %R, %B, %seq_len, %initial_h) {hidden_size = 4 : si64} : (tensor<?x?x10xf32>, tensor<1x12x10xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, tensor<?xi32>, tensor<?x?x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func.func @test_gru_input_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x10xf32>, [[PARAM_1_:%.+]]: tensor<1x12x10xf32>, [[PARAM_2_:%.+]]: tensor<1x12x4xf32>, [[PARAM_3_:%.+]]: tensor<1x24xf32>, [[PARAM_4_:%.+]]: tensor<?xi32>, [[PARAM_5_:%.+]]: tensor<?x?x4xf32>) -> tensor<*xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_5_]]) {axis = 0 : si64, group_id = 3 : si64} : (tensor<?x?x4xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_4_]]) {axis = 0 : si64, group_id = 2 : si64} : (tensor<?xi32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 2 : si64} : (tensor<?x?x10xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_5_]]) {axis = 1 : si64, group_id = 2 : si64} : (tensor<?x?x4xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x?x10xf32>) -> ()
// CHECK:           [[Y_:%.+]], [[VAR_Y_h_:%.+]] = "onnx.GRU"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]]) {direction = "forward", hidden_size = 4 : si64, layout = 0 : si64, linear_before_reset = 0 : si64} : (tensor<?x?x10xf32>, tensor<1x12x10xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, tensor<?xi32>, tensor<?x?x4xf32>) -> (none, tensor<*xf32>)
// CHECK:           return [[VAR_Y_h_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_rnn_input_dims(%X: tensor<?x?x10xf32>, %W: tensor<1x4x10xf32>, %R: tensor<1x4x4xf32>, %B: tensor<1x8xf32>, %seq_len: tensor<?xi32>, %initial_h: tensor<?x?x4xf32>) -> tensor<*xf32> {
  %Y, %Y_h = "onnx.RNN"(%X, %W, %R, %B, %seq_len, %initial_h) {hidden_size = 4 : si64} : (tensor<?x?x10xf32>, tensor<1x4x10xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, tensor<?xi32>, tensor<?x?x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func.func @test_rnn_input_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x10xf32>, [[PARAM_1_:%.+]]: tensor<1x4x10xf32>, [[PARAM_2_:%.+]]: tensor<1x4x4xf32>, [[PARAM_3_:%.+]]: tensor<1x8xf32>, [[PARAM_4_:%.+]]: tensor<?xi32>, [[PARAM_5_:%.+]]: tensor<?x?x4xf32>) -> tensor<*xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_5_]]) {axis = 0 : si64, group_id = 3 : si64} : (tensor<?x?x4xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_4_]]) {axis = 0 : si64, group_id = 2 : si64} : (tensor<?xi32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_5_]]) {axis = 1 : si64, group_id = 2 : si64} : (tensor<?x?x4xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = 2 : si64} : (tensor<?x?x10xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x?x10xf32>) -> ()
// CHECK:           [[Y_:%.+]], [[VAR_Y_h_:%.+]] = "onnx.RNN"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]]) {activations = ["Tanh", "Tanh"], direction = "forward", hidden_size = 4 : si64, layout = 0 : si64} : (tensor<?x?x10xf32>, tensor<1x4x10xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, tensor<?xi32>, tensor<?x?x4xf32>) -> (none, tensor<*xf32>)
// CHECK:           return [[VAR_Y_h_]] : tensor<*xf32>
// CHECK:         } 
}

