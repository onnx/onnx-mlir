// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_softmax_bf16(%arg0 : tensor<10x20x30xbf16>) -> tensor<10x20x30xbf16> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64} : (tensor<10x20x30xbf16>) -> tensor<10x20x30xbf16>
  "func.return"(%0) : (tensor<10x20x30xbf16>) -> ()
}

// CHECK-LABEL:  func.func @test_softmax_bf16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x20x30xbf16>) -> tensor<10x20x30xbf16> {
// CHECK:         [[CST:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    [[EXP:%.+]] = stablehlo.exponential [[PARAM_0_]] : tensor<10x20x30xbf16>
// CHECK-NEXT:    [[REDUCE:%.+]] = stablehlo.reduce([[EXP]] init: [[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<10x20x30xbf16>, tensor<bf16>) -> tensor<10x30xbf16>
// CHECK-NEXT:    [[DENOM:%.+]] = stablehlo.broadcast_in_dim [[REDUCE]], dims = [0, 2] : (tensor<10x30xbf16>) -> tensor<10x20x30xbf16>
// CHECK-NEXT:    [[RES:%.+]] = stablehlo.divide [[EXP]], [[DENOM]] : tensor<10x20x30xbf16>
// CHECK-NEXT:    return [[RES]] : tensor<10x20x30xbf16>

// -----

func.func @test_softmax_f64(%arg0 : tensor<10x20x30xf64>) -> tensor<10x20x30xf64> {
  %0 = "onnx.Softmax"(%arg0) {axis = -1: si64} : (tensor<10x20x30xf64>) -> tensor<10x20x30xf64>
  "func.return"(%0) : (tensor<10x20x30xf64>) -> ()
}

// CHECK-LABEL:  func.func @test_softmax_f64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x20x30xf64>) -> tensor<10x20x30xf64> {
// CHECK:         [[CST:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    [[EXP:%.+]] = stablehlo.exponential [[PARAM_0_]] : tensor<10x20x30xf64>
// CHECK-NEXT:    [[REDUCE:%.+]] = stablehlo.reduce([[EXP]] init: [[CST]]) applies stablehlo.add across dimensions = [2] : (tensor<10x20x30xf64>, tensor<f64>) -> tensor<10x20xf64>
// CHECK-NEXT:    [[DENOM:%.+]] = stablehlo.broadcast_in_dim [[REDUCE]], dims = [0, 1] : (tensor<10x20xf64>) -> tensor<10x20x30xf64>
// CHECK-NEXT:    [[RES:%.+]] = stablehlo.divide [[EXP]], [[DENOM]] : tensor<10x20x30xf64>
// CHECK-NEXT:    return [[RES]] : tensor<10x20x30xf64>
