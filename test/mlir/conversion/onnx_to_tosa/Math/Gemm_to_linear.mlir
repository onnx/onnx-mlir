// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @gemm_to_linear(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
//CHECK-LABEL:  @gemm_to_linear(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4xf32>) -> tensor<1x4xf32>
//CHECK-DAG:    %[[DUMMY_C:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
//CHECK-DAG:    %[[FC_RES:.*]] = "tosa.fully_connected"(%arg0, %arg1, %[[DUMMY_C]])
//CHECK-DAG:    %[[RES:.*]] = "tosa.add"(%[[FC_RES]], %arg2) : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }

// -----

func.func @gemm_to_linear_opt(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x4xf32> {
   %none = "onnx.NoValue"() {value} : () -> none
   %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, none) -> tensor<1x4xf32>
   return %0 : tensor<1x4xf32>
//CHECK-LABEL:  @gemm_to_linear_opt(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x4xf32>
//CHECK-DAG:    "tosa.const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
//CHECK-DAG:    "tosa.fully_connected"(%arg0, %arg1, %1) : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  }
