// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @reduce_mean(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
return %1 : tensor<2x5x1x1xf32>
// CHECK-LABEL:   func.func @reduce_mean(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.reduce_sum %[[VAL_0]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<0.0101010101> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.mul %[[VAL_2]], %[[VAL_3]] {shift = 0 : i8} : (tensor<2x5x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           return %[[VAL_4]] : tensor<2x5x1x1xf32>
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
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<0.00101010106> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_4]], %[[VAL_5]] {shift = 0 : i8} : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK:           return %[[VAL_6]] : tensor<1x1x1x1xf32>
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
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<0.0101010101> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_3]], %[[VAL_4]] {shift = 0 : i8} : (tensor<2x5xf32>, tensor<1x1xf32>) -> tensor<2x5xf32>
// CHECK:           return %[[VAL_5]] : tensor<2x5xf32>
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
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<0.0101010101> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.mul %[[VAL_2]], %[[VAL_3]] {shift = 0 : i8} : (tensor<2x5x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           return %[[VAL_4]] : tensor<2x5x1x1xf32>
}

// -----

func.func @reduce_mean_noop_with_emtpy_axes_one_none_input(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32> {
%none = "onnx.NoValue"() {value} : () -> none
%0 = "onnx.ReduceMean"(%arg0, %none) {noop_with_empty_axes = 1 : si64} : (tensor<2x5x9x11xf32>, none) ->  tensor<2x5x9x11xf32>
return %0 : tensor<2x5x9x11xf32>
// CHECK-LABEL:   func.func @reduce_mean_noop_with_emtpy_axes_one_none_input(
// CHECK-SAME:                                                               %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32> {
// CHECK:           return %[[VAL_0]] : tensor<2x5x9x11xf32>
}

// -----

func.func @test_reducemeanV13(%arg0: tensor<1x32x112x112xf32>) -> tensor<1x32x1x1xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x32x112x112xf32>) -> tensor<1x32x1x1xf32>
  return %0 : tensor<1x32x1x1xf32>
// CHECK-LABEL:  func.func @test_reducemeanV13
// CHECK:           [[VAR_0_:%.+]] = tosa.reduce_sum %arg0 {axis = 2 : i32}
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reduce_sum [[VAR_0_]] {axis = 3 : i32}
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<7.97193861E-5> : tensor<1x1x1x1xf32>}>
// CHECK:           [[VAR_3_:%.+]] = tosa.mul [[VAR_1_]], [[VAR_2_]] {shift = 0 : i8} : (tensor<1x32x1x1xf32>, tensor<1x1x1x1xf32>)
// CHECK:           return [[VAR_3_]] : tensor<1x32x1x1xf32>
}

// -----

func.func @non_constant_axis(%arg0: tensor<3x2x2xf32>, %arg1: tensor<1xi64>) -> tensor<3x2xf32> {
  %0 = "onnx.ReduceMean"(%arg0, %arg1) {keepdims = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
// CHECK-LABEL: non_constant_axis
// CHECK: onnx.ReduceMean
