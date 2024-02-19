// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_softmax_v13(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.Softmax"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK: test_softmax_v13(%[[VAL_0:.*]]: tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_1:.*]] = tosa.exp %[[VAL_0]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 2 : i32} : (tensor<13x21x3xf32>) -> tensor<13x21x1xf32>
// CHECK: %[[VAL_3:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
// CHECK: %[[VAL_4:.*]] = tosa.mul %[[VAL_1]], %[[VAL_3]] {shift = 0 : i8} : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
}

// -----

func.func @test_softmax_v13_axis_one(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK: test_softmax_v13_axis_one(%[[VAL_0:.*]]: tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_1:.*]] = tosa.exp %[[VAL_0]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 1 : i32} : (tensor<13x21x3xf32>) -> tensor<13x1x3xf32>
// CHECK: %[[VAL_3:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<13x1x3xf32>) -> tensor<13x1x3xf32>
// CHECK: %[[VAL_4:.*]] = tosa.mul %[[VAL_1]], %[[VAL_3]] {shift = 0 : i8} : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
}

// -----

func.func @test_softmax_before_v13(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.SoftmaxV11"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK: test_softmax_before_v13(%[[VAL_0:.*]]: tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_1:.*]] = tosa.exp %[[VAL_0]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 1 : i32} : (tensor<13x21x3xf32>) -> tensor<13x1x3xf32>
// CHECK: %[[VAL_3:.*]] = tosa.reduce_sum %[[VAL_2]] {axis = 2 : i32} : (tensor<13x1x3xf32>) -> tensor<13x1x1xf32>
// CHECK: %[[VAL_4:.*]] = tosa.reciprocal %[[VAL_3]] : (tensor<13x1x1xf32>) -> tensor<13x1x1xf32>
// CHECK: %[[VAL_5:.*]] = tosa.mul %[[VAL_1]], %[[VAL_4]] {shift = 0 : i8} : (tensor<13x21x3xf32>, tensor<13x1x1xf32>) -> tensor<13x21x3xf32>
}

// -----

func.func @test_softmax_before_v13_axis_zero(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.SoftmaxV11"(%arg0) {axis = 0 : si64}: (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK: test_softmax_before_v13_axis_zero(%[[VAL_0:.*]]: tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_1:.*]] = tosa.exp %[[VAL_0]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
// CHECK: %[[VAL_3:.*]] = tosa.reduce_sum %[[VAL_2]] {axis = 1 : i32} : (tensor<1x21x3xf32>) -> tensor<1x1x3xf32>
// CHECK: %[[VAL_4:.*]] = tosa.reduce_sum %[[VAL_3]] {axis = 2 : i32} : (tensor<1x1x3xf32>) -> tensor<1x1x1xf32>
// CHECK: %[[VAL_5:.*]] = tosa.reciprocal %[[VAL_4]] : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
// CHECK: %[[VAL_6:.*]] = tosa.mul %[[VAL_1]], %[[VAL_5]] {shift = 0 : i8} : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
}