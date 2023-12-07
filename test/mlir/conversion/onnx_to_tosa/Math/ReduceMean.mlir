// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @reduce_mean(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
return %1 : tensor<2x5x1x1xf32>
// CHECK-LABEL:   func.func @reduce_mean(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.reduce_sum %[[VAL_0]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<0.0101010101> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_3]] {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<f32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_2]], %[[VAL_4]] {shift = 0 : i8} : (tensor<2x5x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           return %[[VAL_5]] : tensor<2x5x1x1xf32>
}

// -----

func.func @reduce_mean_no_axes_attr(%arg0: tensor<2x5x9x11xf32>) -> tensor<1x1x1x1xf32> {
%none = "onnx.NoValue"() {value} : () -> none
%0 = "onnx.ReduceMean"(%arg0, %none) : (tensor<2x5x9x11xf32>, none) -> tensor<1x1x1x1xf32>
return %0 : tensor<1x1x1x1xf32>
// CHECK-LABEL:   func.func @reduce_mean_no_axes_attr(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<1x1x1x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.reduce_sum %[[VAL_0]] {axis = 0 : i32} : (tensor<2x5x9x11xf32>) -> tensor<1x5x9x11xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 1 : i32} : (tensor<1x5x9x11xf32>) -> tensor<1x1x9x11xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.reduce_sum %[[VAL_2]] {axis = 2 : i32} : (tensor<1x1x9x11xf32>) -> tensor<1x1x1x11xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reduce_sum %[[VAL_3]] {axis = 3 : i32} : (tensor<1x1x1x11xf32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<0.00101010106> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.reshape %[[VAL_5]] {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<f32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.mul %[[VAL_4]], %[[VAL_6]] {shift = 0 : i8} : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK:           return %[[VAL_7]] : tensor<1x1x1x1xf32>
}

// -----

func.func @reduce_mean_keepdims_false(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5xf32>
return %1 : tensor<2x5xf32>
// CHECK-LABEL:   func.func @reduce_mean_keepdims_false(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.reduce_sum %[[VAL_0]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.reshape %[[VAL_2]] {new_shape = array<i64: 2, 5>} : (tensor<2x5x1x1xf32>) -> tensor<2x5xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<0.0101010101> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_4]] {new_shape = array<i64: 1, 1>} : (tensor<f32>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_3]], %[[VAL_5]] {shift = 0 : i8} : (tensor<2x5xf32>, tensor<1x1xf32>) -> tensor<2x5xf32>
// CHECK:           return %[[VAL_6]] : tensor<2x5xf32>
}

// -----

func.func @reduce_mean_noop_with_emtpy_axes_one(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) {noop_with_empty_axes = 1 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
return %1 : tensor<2x5x1x1xf32>
// CHECK-LABEL:   func.func @reduce_mean_noop_with_emtpy_axes_one(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.reduce_sum %[[VAL_0]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<0.0101010101> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_3]] {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<f32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_2]], %[[VAL_4]] {shift = 0 : i8} : (tensor<2x5x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           return %[[VAL_5]] : tensor<2x5x1x1xf32>
}

// -----

func.func @reduce_mean_noop_with_emtpy_axes_one_none_input(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32> {
%none = "onnx.NoValue"() {value} : () -> none
%0 = "onnx.ReduceMean"(%arg0, %none) {noop_with_empty_axes = 1 : si64} : (tensor<2x5x9x11xf32>, none) ->  tensor<2x5x9x11xf32>
return %0 : tensor<2x5x9x11xf32>
// CHECK-LABEL:   func.func @reduce_mean_noop_with_emtpy_axes_one_none_input(
// CHECK-SAME:                                                               %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.identity %[[VAL_0]] : (tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_3:.*]] = tosa.reshape %[[VAL_2]] {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<f32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.mul %[[VAL_1]], %[[VAL_3]] {shift = 0 : i8} : (tensor<2x5x9x11xf32>, tensor<1x1x1x1xf32>) -> tensor<2x5x9x11xf32>
// CHECK:           return %[[VAL_4]] : tensor<2x5x9x11xf32>
}
