// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @reduce_mean(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3]} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32>
return %0 : tensor<2x5x1x1xf32>
// CHECK-LABEL: func @reduce_mean
// CHECK: [[VAR0:%.*]] = "tosa.reduce_sum"(%arg0) {axis = 2 : i64} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK: [[VAR1:%.*]] = "tosa.reduce_sum"([[VAR0]]) {axis = 3 : i64} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK: [[VAR2:%.*]] = "tosa.const"() {value = dense<0.0101010101> : tensor<f32>} : () -> tensor<f32>
// CHECK: {{%.*}} = "tosa.mul"([[VAR1]], [[VAR2]]) {shift = 0 : i32} : (tensor<2x5x1x1xf32>, tensor<f32>) -> tensor<2x5x1x1xf32>
}

// -----
func.func @reduce_mean_no_axes_attr(%arg0: tensor<2x5x9x11xf32>) -> tensor<1x1x1x1xf32> {
%0 = "onnx.ReduceMeanV13"(%arg0) : (tensor<2x5x9x11xf32>) -> tensor<1x1x1x1xf32>
return %0 : tensor<1x1x1x1xf32>
// CHECK-LABEL: func @reduce_mean_no_axes_attr
// CHECK: [[VAR0:%.*]] = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<2x5x9x11xf32>) -> tensor<1x5x9x11xf32>
// CHECK: [[VAR1:%.*]] = "tosa.reduce_sum"([[VAR0]]) {axis = 1 : i64} : (tensor<1x5x9x11xf32>) -> tensor<1x1x9x11xf32>
// CHECK: [[VAR2:%.*]] = "tosa.reduce_sum"([[VAR1]]) {axis = 2 : i64} : (tensor<1x1x9x11xf32>) -> tensor<1x1x1x11xf32>
// CHECK: [[VAR3:%.*]] = "tosa.reduce_sum"([[VAR2]]) {axis = 3 : i64} : (tensor<1x1x1x11xf32>) -> tensor<1x1x1x1xf32>
// CHECK: [[VAR4:%.*]] = "tosa.const"() {value = dense<0.00101010106> : tensor<f32>} : () -> tensor<f32>
// CHECK: {{%.*}} = "tosa.mul"([[VAR3]], [[VAR4]]) {shift = 0 : i32} : (tensor<1x1x1x1xf32>, tensor<f32>) -> tensor<1x1x1x1xf32>
}

// -----
func.func @reduce_mean_keepdims_false(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
%0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 0 : si64} : (tensor<2x5x9x11xf32>) -> tensor<2x5xf32>
return %0 : tensor<2x5xf32>
// CHECK-LABEL: func @reduce_mean_keepdims_false
// CHECK: [[VAR0:%.*]] = "tosa.reduce_sum"(%arg0) {axis = 2 : i64} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK: [[VAR1:%.*]] = "tosa.reduce_sum"([[VAR0]]) {axis = 3 : i64} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK: [[VAR2:%.*]] = "tosa.reshape"([[VAR1]]) {new_shape = array<i64: 2, 5>} : (tensor<2x5x1x1xf32>) -> tensor<2x5xf32>
// CHECK: [[VAR3:%.*]] = "tosa.const"() {value = dense<0.0101010101> : tensor<f32>} : () -> tensor<f32>
// CHECK: {{%.*}} = "tosa.mul"([[VAR2]], [[VAR3]]) {shift = 0 : i32} : (tensor<2x5xf32>, tensor<f32>) ->  tensor<2x5xf32>
}

