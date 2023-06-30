// RUN: onnx-mlir-opt --convert-onnx-to-tosa %s -split-input-file | FileCheck %s


func.func @test_concat(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) { axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32>
  "func.return"(%0) : (tensor<5x5x4x32xf32>) -> ()
// CHECK-LABEL:   func.func @test_concat(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<5x5x1x32xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_1]]) <{axis = 2 : i64}> : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32>
// CHECK:           return %[[VAL_2]] : tensor<5x5x4x32xf32>
}

// -----
func.func @test_concat_dynamic_shape(%arg0 : tensor<5x5x?x32xf32>, %arg1 : tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) { axis = 2 : si64} : (tensor<5x5x?x32xf32>, tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32>
  "func.return"(%0) : (tensor<5x5x?x32xf32>) -> ()
// CHECK-LABEL:   func.func @test_concat_dynamic_shape(
// CHECK-SAME:                                         %[[VAL_0:.*]]: tensor<5x5x?x32xf32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_1]]) <{axis = 2 : i64}> : (tensor<5x5x?x32xf32>, tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32>
// CHECK:           return %[[VAL_2]] : tensor<5x5x?x32xf32>
}

// -----
func.func @test_concat_negative_axis(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) { axis = -2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32>
  "func.return"(%0) : (tensor<5x5x4x32xf32>) -> ()
// CHECK-LABEL:   func.func @test_concat_negative_axis(
// CHECK-SAME:                                         %[[VAL_0:.*]]: tensor<5x5x1x32xf32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_1]]) <{axis = 2 : i64}> : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32>
// CHECK:           return %[[VAL_2]] : tensor<5x5x4x32xf32>
}
