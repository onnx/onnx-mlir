// RUN: onnx-mlir-opt --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_eyelike_dtype_f32(%arg0 : tensor<4x4xi32>) -> tensor<4x4xf32> {
  %1 = "onnx.EyeLike"(%arg0) {dtype = 1 : si64} : (tensor<4x4xi32>) -> tensor<4x4xf32>
  "onnx.Return"(%1) : (tensor<4x4xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_eyelike_dtype_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x4xi32>) -> tensor<4x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<4x4xf32>}> : () -> tensor<4x4xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<4x4xf32>

// -----

func.func @test_eyelike_int8(%arg0 : tensor<4x4xi8>) -> tensor<4x4xi8> {
  %1 = "onnx.EyeLike"(%arg0) : (tensor<4x4xi8>) -> tensor<4x4xi8>
  "onnx.Return"(%1) : (tensor<4x4xi8>) -> ()
}
// CHECK-LABEL:  func.func @test_eyelike_int8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x4xi8>) -> tensor<4x4xi8> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{\[\[}}1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]> : tensor<4x4xi8>}> : () -> tensor<4x4xi8>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<4x4xi8>

// -----

func.func @test_eyelike_bool(%arg0 : tensor<4x4xi1>) -> tensor<4x4xi1> {
  %1 = "onnx.EyeLike"(%arg0) : (tensor<4x4xi1>) -> tensor<4x4xi1>
  "onnx.Return"(%1) : (tensor<4x4xi1>) -> ()
}
// CHECK-LABEL:  func.func @test_eyelike_bool
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x4xi1>) -> tensor<4x4xi1> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{\[\[}}true, false, false, false], [false, true, false, false], [false, false, true, false], [false, false, false, true]]> : tensor<4x4xi1>}> : () -> tensor<4x4xi1>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<4x4xi1>

// -----

func.func @test_eyelike_k_pos(%arg0 : tensor<4x4xf64>) -> tensor<4x4xf64> {
  %1 = "onnx.EyeLike"(%arg0) {k = 2 : si64} : (tensor<4x4xf64>) -> tensor<4x4xf64>
  "onnx.Return"(%1) : (tensor<4x4xf64>) -> ()
}
// CHECK-LABEL:  func.func @test_eyelike_k_pos
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x4xf64>) -> tensor<4x4xf64> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{\[\[}}0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<4x4xf64>}> : () -> tensor<4x4xf64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<4x4xf64>

// -----

func.func @test_eyelike_k_neg(%arg0 : tensor<4x4xf64>) -> tensor<4x4xf64> {
  %1 = "onnx.EyeLike"(%arg0) {k = -2 : si64} : (tensor<4x4xf64>) -> tensor<4x4xf64>
  "onnx.Return"(%1) : (tensor<4x4xf64>) -> ()
}
// CHECK-LABEL:  func.func @test_eyelike_k_neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x4xf64>) -> tensor<4x4xf64> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{\[\[}}0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<4x4xf64>}> : () -> tensor<4x4xf64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<4x4xf64>

// -----

func.func @test_eyelike_k_out_of_rang(%arg0 : tensor<4x4xf64>) -> tensor<4x4xf64> {
  %1 = "onnx.EyeLike"(%arg0) {k = 42 : si64} : (tensor<4x4xf64>) -> tensor<4x4xf64>
  "onnx.Return"(%1) : (tensor<4x4xf64>) -> ()
}
// CHECK-LABEL:  func.func @test_eyelike_k_out_of_rang
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x4xf64>) -> tensor<4x4xf64> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<4x4xf64>}> : () -> tensor<4x4xf64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<4x4xf64>

// -----

func.func @test_eyelike_dif_dim(%arg0 : tensor<2x5xf64>) -> tensor<2x5xf64> {
  %1 = "onnx.EyeLike"(%arg0)  : (tensor<2x5xf64>) -> tensor<2x5xf64>
  "onnx.Return"(%1) : (tensor<2x5xf64>) -> ()
}
// CHECK-LABEL:  func.func @test_eyelike_dif_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5xf64>) -> tensor<2x5xf64> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<2x5xf64>}> : () -> tensor<2x5xf64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x5xf64>

// -----

func.func @test_eyelike_dif_dim_k(%arg0 : tensor<2x5xf64>) -> tensor<2x5xf64> {
  %1 = "onnx.EyeLike"(%arg0) {k = 1 : si64}  : (tensor<2x5xf64>) -> tensor<2x5xf64>
  "onnx.Return"(%1) : (tensor<2x5xf64>) -> ()
}
// CHECK-LABEL:  func.func @test_eyelike_dif_dim_k
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5xf64>) -> tensor<2x5xf64> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{\[\[}}0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<2x5xf64>}> : () -> tensor<2x5xf64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x5xf64>

// -----

func.func @test_eyelike_dif_dim_2(%arg0 : tensor<4x2xf64>) -> tensor<4x2xf64> {
  %1 = "onnx.EyeLike"(%arg0)  : (tensor<4x2xf64>) -> tensor<4x2xf64>
  "onnx.Return"(%1) : (tensor<4x2xf64>) -> ()
}
// CHECK-LABEL:  func.func @test_eyelike_dif_dim_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x2xf64>) -> tensor<4x2xf64> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{\[\[}}1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00], [0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]{{.}}> : tensor<4x2xf64>}> : () -> tensor<4x2xf64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<4x2xf64>
