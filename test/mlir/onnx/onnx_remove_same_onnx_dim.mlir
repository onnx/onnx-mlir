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

// -----

// onnx.Dim of intermediate tensors are replaced by onnx.Dim of a function
// argument that has the same dynamic dimension. The existing onnx.Dim of the
// argument (axis 1) is reused, and a new one is created for axis 0.

func.func @test_dim_of_func_arg(%arg0: tensor<?x?xi64> {onnx.dim_params = "0:batch_size,1:sequence_length"}, %arg1: tensor<30522x768xf32>) -> tensor<4xi64> {
  %c = onnx.Constant dense<12> : tensor<1xi64>
  %s = "onnx.Dim"(%arg0) <{axis = 1 : si64}> : (tensor<?x?xi64>) -> tensor<1xi64>
  %g = "onnx.Gather"(%arg1, %arg0) <{axis = 0 : si64}> : (tensor<30522x768xf32>, tensor<?x?xi64>) -> tensor<?x?x768xf32>
  %r = "onnx.Relu"(%g) : (tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
  %d0 = "onnx.Dim"(%r) <{axis = 0 : si64}> : (tensor<?x?x768xf32>) -> tensor<1xi64>
  %d1 = "onnx.Dim"(%r) <{axis = 1 : si64}> : (tensor<?x?x768xf32>) -> tensor<1xi64>
  %cc = "onnx.Concat"(%d0, %d1, %s, %c) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  onnx.Return %cc : tensor<4xi64>

// CHECK-LABEL:  func.func @test_dim_of_func_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xi64> {onnx.dim_params = "0:batch_size,1:sequence_length"}, [[PARAM_1_:%.+]]: tensor<30522x768xf32>) -> tensor<4xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) <{axis = 1 : si64}> : (tensor<?x?xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) <{axis = 0 : si64}> : (tensor<?x?xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Gather"([[PARAM_1_]], [[PARAM_0_]]) <{axis = 0 : si64}> : (tensor<30522x768xf32>, tensor<?x?xi64>) -> tensor<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Relu"([[VAR_3_]]) : (tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_0_]], [[VAR_0_]], [[VAR_2_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           onnx.Return [[VAR_5_]] : tensor<4xi64>
// CHECK:         }
}

// -----

// No onnx.Dim of the function arguments exists: create them at the beginning
// of the function, using the lowest argument number when several arguments
// have the same dynamic dimension.

func.func @test_create_dim_of_func_arg(%arg0: tensor<?x?xf32> {onnx.dim_params = "0:M,1:N"}, %arg1: tensor<?x?xf32> {onnx.dim_params = "0:M,1:N"}) -> tensor<2xi64> {
  %a = "onnx.Add"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %r = "onnx.Relu"(%a) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %d0 = "onnx.Dim"(%r) <{axis = 0 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
  %d1 = "onnx.Dim"(%r) <{axis = 1 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
  %cc = "onnx.Concat"(%d0, %d1) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  onnx.Return %cc : tensor<2xi64>

// CHECK-LABEL:  func.func @test_create_dim_of_func_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32> {onnx.dim_params = "0:M,1:N"}, [[PARAM_1_:%.+]]: tensor<?x?xf32> {onnx.dim_params = "0:M,1:N"}) -> tensor<2xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Dim"([[PARAM_0_]]) <{axis = 1 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) <{axis = 0 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Relu"([[VAR_2_]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_0_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<2xi64>
// CHECK:         }
}

// -----

// Dynamic dimensions not related to function arguments keep the old behavior:
// later onnx.Dim are replaced by the first one.

func.func @test_no_func_arg_dim(%arg0: tensor<?xi64>) -> tensor<2xi64> {
  %s = onnx.Constant dense<[2, -1]> : tensor<2xi64>
  %r = "onnx.Reshape"(%arg0, %s) <{allowzero = 0 : si64}> : (tensor<?xi64>, tensor<2xi64>) -> tensor<2x?xi64>
  %n = "onnx.Neg"(%r) : (tensor<2x?xi64>) -> tensor<2x?xi64>
  %d0 = "onnx.Dim"(%r) <{axis = 1 : si64}> : (tensor<2x?xi64>) -> tensor<1xi64>
  %d1 = "onnx.Dim"(%n) <{axis = 1 : si64}> : (tensor<2x?xi64>) -> tensor<1xi64>
  %cc = "onnx.Concat"(%d0, %d1) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  onnx.Return %cc : tensor<2xi64>

// CHECK-LABEL:  func.func @test_no_func_arg_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<2xi64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, -1]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<?xi64>, tensor<2xi64>) -> tensor<2x?xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Neg"([[VAR_1_]]) : (tensor<2x?xi64>) -> tensor<2x?xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Dim"([[VAR_1_]]) <{axis = 1 : si64}> : (tensor<2x?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_3_]], [[VAR_3_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<2xi64>
// CHECK:         }
}
