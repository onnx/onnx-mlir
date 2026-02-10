// RUN: onnx-mlir-opt --remove-same-onnx-dim %s -split-input-file | FileCheck %s

func.func @test_dim_params_onnx_return(%arg0: tensor<?x?xf32> {onnx.dim_params = "0:M,1:N", onnx.name = "X"}, %arg1: tensor<?x?xf32> {onnx.dim_params = "0:M,1:P", onnx.name = "Y"}) -> (tensor<1xi64>) {
  %M = "onnx.Dim"(%arg0) <{axis = 0 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
  %N = "onnx.Dim"(%arg0) <{axis = 1 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%N, %M) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %reshape = "onnx.Reshape"(%arg0, %shape) : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  %dim_0 = "onnx.Dim"(%reshape) <{axis = 0 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
  %dim_1 = "onnx.Dim"(%reshape) <{axis = 1 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
  %output = "onnx.Mul"(%dim_0, %dim_1) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  onnx.Return %output: tensor<1xi64>

// CHECK-LABEL:  func.func @test_dim_params_onnx_return
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32> {onnx.dim_params = "0:M,1:N", onnx.name = "X"}, [[PARAM_1_:%.+]]: tensor<?x?xf32> {onnx.dim_params = "0:M,1:P", onnx.name = "Y"}) -> tensor<1xi64> {
  // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) <{axis = 0 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
  // CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) <{axis = 1 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
  // CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_0_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  // CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  // CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_1_]], [[VAR_0_]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  // CHECK:           onnx.Return [[VAR_4_]] : tensor<1xi64>
  // CHECK:         }
}
