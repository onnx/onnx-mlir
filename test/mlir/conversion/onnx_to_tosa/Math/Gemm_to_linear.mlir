// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @gemm_to_fc(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
// CHECK-LABEL:   func.func @gemm_to_fc(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<1x5xf32>,
// CHECK-SAME:                          %[[VAL_1:.*]]: tensor<4x5xf32>,
// CHECK-SAME:                          %[[VAL_2:.*]]: tensor<4xf32>) -> tensor<1x4xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.fully_connected"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
// CHECK:           return %[[VAL_3]] : tensor<1x4xf32>
// CHECK:         }
}
  
// -----
  
func.func @gemm_to_fc_broadcast(%arg0: tensor<2x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<2x5xf32>, tensor<4x5xf32>, tensor<1xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
// CHECK-LABEL:   func.func @gemm_to_fc_broadcast(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<2x5xf32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: tensor<4x5xf32>,
// CHECK-SAME:                                    %[[VAL_2:.*]]: tensor<1xf32>) -> tensor<2x4xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.fully_connected"(%[[VAL_0]], %[[VAL_1]], %[[VAL_3]]) : (tensor<2x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.add"(%[[VAL_4]], %[[VAL_2]]) : (tensor<2x4xf32>, tensor<1xf32>) -> tensor<2x4xf32>
// CHECK:           return %[[VAL_5]] : tensor<2x4xf32>
// CHECK:         }
}

// -----

func.func @gemm_to_fc_opt(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, none) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
// CHECK-LABEL:   func.func @gemm_to_fc_opt(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<1x5xf32>,
// CHECK-SAME:                              %[[VAL_1:.*]]: tensor<4x5xf32>) -> tensor<1x4xf32> {
// CHECK:           %[[VAL_2:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.fully_connected"(%[[VAL_0]], %[[VAL_1]], %[[VAL_3]]) : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x4xf32>
// CHECK:         }
}
