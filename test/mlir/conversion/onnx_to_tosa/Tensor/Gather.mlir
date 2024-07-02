// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_gather_axis0(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()
// CHECK-LABEL:   func.func @test_gather_axis0(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<3x2xf32>) -> tensor<2x2x2xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{value = dense<{{\[\[}}0, 1], [1, 2]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_3:.*]] = tosa.add %[[VAL_1]], %[[VAL_2]] : (tensor<2x2xi64>, tensor<1x1xi64>) -> tensor<2x2xi64>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_5:.*]] = tosa.greater_equal %[[VAL_1]], %[[VAL_4]] : (tensor<2x2xi64>, tensor<1x1xi64>) -> tensor<2x2xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.select %[[VAL_5]], %[[VAL_1]], %[[VAL_3]] : (tensor<2x2xi1>, tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
// CHECK:           %[[VAL_7:.*]] = tosa.cast %[[VAL_6]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{value = dense<[0, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_8]] : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<3x2xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]] {new_shape = array<i64: 1, 3, 2>} : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_7]] {new_shape = array<i64: 1, 4>} : (tensor<2x2xi32>) -> tensor<1x4xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.gather %[[VAL_10]], %[[VAL_11]] : (tensor<1x3x2xf32>, tensor<1x4xi32>) -> tensor<1x4x2xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_12]] {new_shape = array<i64: 2, 2, 2>} : (tensor<1x4x2xf32>) -> tensor<2x2x2xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{value = dense<[0, 1, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_15:.*]] = tosa.transpose %[[VAL_13]], %[[VAL_14]] : (tensor<2x2x2xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
// CHECK:           return %[[VAL_15]] : tensor<2x2x2xf32>
}

// -----

// Test negative indices.
func.func @test_gather_axis0_neg_idx(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()
// CHECK-LABEL:   func.func @test_gather_axis0_neg_idx(
// CHECK-SAME:                                         %[[VAL_0:.*]]: tensor<3x2xf32>) -> tensor<2x2x2xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{value = dense<{{\[\[}}0, -1], [1, 2]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_3:.*]] = tosa.add %[[VAL_1]], %[[VAL_2]] : (tensor<2x2xi64>, tensor<1x1xi64>) -> tensor<2x2xi64>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_5:.*]] = tosa.greater_equal %[[VAL_1]], %[[VAL_4]] : (tensor<2x2xi64>, tensor<1x1xi64>) -> tensor<2x2xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.select %[[VAL_5]], %[[VAL_1]], %[[VAL_3]] : (tensor<2x2xi1>, tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
// CHECK:           %[[VAL_7:.*]] = tosa.cast %[[VAL_6]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{value = dense<[0, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_8]] : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<3x2xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]] {new_shape = array<i64: 1, 3, 2>} : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_7]] {new_shape = array<i64: 1, 4>} : (tensor<2x2xi32>) -> tensor<1x4xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.gather %[[VAL_10]], %[[VAL_11]] : (tensor<1x3x2xf32>, tensor<1x4xi32>) -> tensor<1x4x2xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_12]] {new_shape = array<i64: 2, 2, 2>} : (tensor<1x4x2xf32>) -> tensor<2x2x2xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{value = dense<[0, 1, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_15:.*]] = tosa.transpose %[[VAL_13]], %[[VAL_14]] : (tensor<2x2x2xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
// CHECK:           return %[[VAL_15]] : tensor<2x2x2xf32>
}

// -----

// Test along axis 1. Transpose should be different.
func.func @test_gather_axis1(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 2]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()
// CHECK-LABEL:   func.func @test_gather_axis1(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<3x3xf32>) -> tensor<3x1x2xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{value = dense<{{\[\[}}0, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_3:.*]] = tosa.add %[[VAL_1]], %[[VAL_2]] : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi64>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_5:.*]] = tosa.greater_equal %[[VAL_1]], %[[VAL_4]] : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.select %[[VAL_5]], %[[VAL_1]], %[[VAL_3]] : (tensor<1x2xi1>, tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<1x2xi64>
// CHECK:           %[[VAL_7:.*]] = tosa.cast %[[VAL_6]] : (tensor<1x2xi64>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_8]] : (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]] {new_shape = array<i64: 1, 3, 3>} : (tensor<3x3xf32>) -> tensor<1x3x3xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_7]] {new_shape = array<i64: 1, 2>} : (tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.gather %[[VAL_10]], %[[VAL_11]] : (tensor<1x3x3xf32>, tensor<1x2xi32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_12]] {new_shape = array<i64: 1, 2, 3>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{value = dense<[2, 0, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_15:.*]] = tosa.transpose %[[VAL_13]], %[[VAL_14]] : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x1x2xf32>
// CHECK:           return %[[VAL_15]] : tensor<3x1x2xf32>
// CHECK:         }
}

// -----

func.func @test_gather_dynamic_indices(%arg0 : tensor<3x3xf32>, %indices: tensor<1x2xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()
// CHECK-LABEL:   func.func @test_gather_dynamic_indices(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<3x3xf32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: tensor<1x2xi64>) -> tensor<3x1x2xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_3:.*]] = tosa.add %[[VAL_1]], %[[VAL_2]] : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi64>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_5:.*]] = tosa.greater_equal %[[VAL_1]], %[[VAL_4]] : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.select %[[VAL_5]], %[[VAL_1]], %[[VAL_3]] : (tensor<1x2xi1>, tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<1x2xi64>
// CHECK:           %[[VAL_7:.*]] = tosa.cast %[[VAL_6]] : (tensor<1x2xi64>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_8]] : (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]] {new_shape = array<i64: 1, 3, 3>} : (tensor<3x3xf32>) -> tensor<1x3x3xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_7]] {new_shape = array<i64: 1, 2>} : (tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.gather %[[VAL_10]], %[[VAL_11]] : (tensor<1x3x3xf32>, tensor<1x2xi32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_12]] {new_shape = array<i64: 1, 2, 3>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{value = dense<[2, 0, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_15:.*]] = tosa.transpose %[[VAL_13]], %[[VAL_14]] : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x1x2xf32>
// CHECK:           return %[[VAL_15]] : tensor<3x1x2xf32>
}

// -----

func.func @test_gather_dynamic_indices_i32(%arg0 : tensor<3x3xf32>, %indices: tensor<1x2xi32>) -> tensor<3x1x2xf32> {
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi32>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()
// CHECK-LABEL:   func.func @test_gather_dynamic_indices_i32(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<3x3xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<1x2xi32>) -> tensor<3x1x2xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_3:.*]] = tosa.cast %[[VAL_2]] : (tensor<1x1xi64>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_4:.*]] = tosa.add %[[VAL_1]], %[[VAL_3]] : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_6:.*]] = tosa.cast %[[VAL_5]] : (tensor<1x1xi64>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.greater_equal %[[VAL_1]], %[[VAL_6]] : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi1>
// CHECK:           %[[VAL_8:.*]] = tosa.select %[[VAL_7]], %[[VAL_1]], %[[VAL_4]] : (tensor<1x2xi1>, tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK:           %[[VAL_10:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_9]] : (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_10]] {new_shape = array<i64: 1, 3, 3>} : (tensor<3x3xf32>) -> tensor<1x3x3xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.reshape %[[VAL_8]] {new_shape = array<i64: 1, 2>} : (tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_13:.*]] = tosa.gather %[[VAL_11]], %[[VAL_12]] : (tensor<1x3x3xf32>, tensor<1x2xi32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.reshape %[[VAL_13]] {new_shape = array<i64: 1, 2, 3>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_15:.*]] = "tosa.const"() <{value = dense<[2, 0, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_16:.*]] = tosa.transpose %[[VAL_14]], %[[VAL_15]] : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x1x2xf32>
// CHECK:           return %[[VAL_16]] : tensor<3x1x2xf32>
}

// -----

func.func @test_gather_like_slice(%arg0 : tensor<3x3xf32>) -> tensor<3xf32> {
  %indices = onnx.Constant dense<0> : tensor<i64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<i64>) -> tensor<3xf32>
  "func.return"(%0) : (tensor<3xf32>) -> ()
// CHECK-LABEL:   func.func @test_gather_like_slice(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<3x3xf32>) -> tensor<3x1x2xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{value = dense<{{\[\[}}0, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_3:.*]] = tosa.add %[[VAL_1]], %[[VAL_2]] : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi64>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_5:.*]] = tosa.greater_equal %[[VAL_1]], %[[VAL_4]] : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.select %[[VAL_5]], %[[VAL_1]], %[[VAL_3]] : (tensor<1x2xi1>, tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<1x2xi64>
// CHECK:           %[[VAL_7:.*]] = tosa.cast %[[VAL_6]] : (tensor<1x2xi64>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_8]] : (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]] {new_shape = array<i64: 1, 3, 3>} : (tensor<3x3xf32>) -> tensor<1x3x3xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_7]] {new_shape = array<i64: 1, 2>} : (tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.gather %[[VAL_10]], %[[VAL_11]] : (tensor<1x3x3xf32>, tensor<1x2xi32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_12]] {new_shape = array<i64: 1, 2, 3>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{value = dense<[2, 0, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_15:.*]] = tosa.transpose %[[VAL_13]], %[[VAL_14]] : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x1x2xf32>
// CHECK:           return %[[VAL_15]] : tensor<3x1x2xf32>
// CHECK:         }
}
