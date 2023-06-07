// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

/// Test the default behavior of AveragePool with no padding
func.func @test_default_averagepool(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  "func.return"(%0) : (tensor<5x5x30x30xf32>) -> ()
}
// CHECK-LABEL: func.func @test_default_averagepool(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
// CHECK-DAG:   "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:   %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:   %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x32x32x5xf32>) -> tensor<5x30x30x5xf32>
// CHECK-DAG:   "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:   %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x30x30x5xf32>, tensor<4xi32>) -> tensor<5x5x30x30xf32>
// CHECK-DAG:   return %[[TRANS_MPOOL_RES]] : tensor<5x5x30x30xf32>

// -----

/// Test the behavior of AveragePool with uniform padding
func.func @test_default_averagepool_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [3,3], pads = [1, 1, 1, 1] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
  "func.return"(%0) : (tensor<5x5x32x32xf32>) -> ()
}
// CHECK-DAG:   func.func @test_default_averagepool_pad(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<5x32x32x5xf32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x32x32x5xf32>, tensor<4xi32>) -> tensor<5x5x32x32xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x32x32xf32>

// -----

/// Test the behavior of AveragePool with non uniform padding
func.func @test_default_averagepool_pad_nonunif(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32> {
  %0 = "onnx.AveragePool"(%arg0) {ceil_mode = 0 : si64, kernel_shape = [5,3], pads = [0, 1, 2, 3] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32>
  "func.return"(%0) : (tensor<5x5x30x34xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_default_averagepool_pad_nonunif(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32> {
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:    %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:    %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = array<i64: 5, 3>, pad = array<i64: 0, 2, 1, 3>, stride = array<i64: 1, 1>} : (tensor<5x32x32x5xf32>) -> tensor<5x30x34x5xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:    %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x30x34x5xf32>, tensor<4xi32>) -> tensor<5x5x30x34xf32>
// CHECK-DAG:    return %[[TRANS_MPOOL_RES]] : tensor<5x5x30x34xf32>
// -----

/// Test the behavior of AveragePool with strides set
func.func @test_default_averagepool_strides(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [3,3], pads = [1, 1, 1, 1], strides = [2, 2] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32>
  "func.return"(%0) : (tensor<5x5x16x16xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_default_averagepool_strides(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<5x32x32x5xf32>) -> tensor<5x16x16x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x16x16x5xf32>, tensor<4xi32>) -> tensor<5x5x16x16xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x16x16xf32>

// -----

/// Test the behavior of AveragePool with strides and non uniform padding 
func.func @test_default_averagepool_strides_nonunifpad(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32>
  "func.return"(%0) : (tensor<5x5x15x16xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_default_averagepool_strides_nonunifpad(%arg0: tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x30x32xf32>, tensor<4xi32>) -> tensor<5x30x32x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = array<i64: 2, 2>, pad = array<i64: 1, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<5x30x32x5xf32>) -> tensor<5x15x16x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x15x16x5xf32>, tensor<4xi32>) -> tensor<5x5x15x16xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x15x16xf32>

// -----

/// Test the behavior of AveragePool with ceiling set (Should change the result shape)
func.func @test_default_averagepool_strides_nonunifpad_ceil(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32> {
  %0 = "onnx.AveragePool"(%arg0) {ceil_mode = 1 : si64, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32>
  "func.return"(%0) : (tensor<5x5x16x16xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_default_averagepool_strides_nonunifpad_ceil(%arg0: tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x30x32xf32>, tensor<4xi32>) -> tensor<5x30x32x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = array<i64: 2, 2>, pad = array<i64: 1, 2, 0, 0>, stride = array<i64: 2, 2>} : (tensor<5x30x32x5xf32>) -> tensor<5x16x16x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x16x16x5xf32>, tensor<4xi32>) -> tensor<5x5x16x16xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x16x16xf32>


// -----

func.func @test_default_averagepool_autopad_valid(%arg0 : tensor<5x5x16x13xf32>) -> tensor<5x5x14x11xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "VALID", kernel_shape = [3,3]} : (tensor<5x5x16x13xf32>) -> tensor<5x5x14x11xf32>
  "func.return"(%0) : (tensor<5x5x14x11xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_default_averagepool_autopad_valid(%arg0: tensor<5x5x16x13xf32>) -> tensor<5x5x14x11xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x16x13xf32>, tensor<4xi32>) -> tensor<5x16x13x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x16x13x5xf32>) -> tensor<5x14x11x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x14x11x5xf32>, tensor<4xi32>) -> tensor<5x5x14x11xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x14x11xf32>

// -----

func.func @test_default_averagepool_same_upper_ceil_mode(%arg0 : tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "SAME_UPPER", ceil_mode = 1 : si64, kernel_shape = [4,4], strides = [4, 4] } : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
  "func.return"(%0) : (tensor<5x5x4x4xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_default_averagepool_same_upper_ceil_mode(%arg0: tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x16x13xf32>, tensor<4xi32>) -> tensor<5x16x13x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = array<i64: 4, 4>, pad = array<i64: 0, 0, 1, 2>, stride = array<i64: 4, 4>} : (tensor<5x16x13x5xf32>) -> tensor<5x4x4x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x4x4x5xf32>, tensor<4xi32>) -> tensor<5x5x4x4xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x4x4xf32>

// -----

/// Test the behavior of AveragePool with uniform padding and count_include_pad == 1
func.func @test_averagepool_pad_with_count_include_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
  %0 = "onnx.AveragePool"(%arg0) {count_include_pad = 1 : si64, kernel_shape = [3,3], pads = [1, 1, 1, 1] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
  "func.return"(%0) : (tensor<5x5x32x32xf32>) -> ()
}
// CHECK-DAG:   func.func @test_averagepool_pad_with_count_include_pad(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
// CHECK-DAG:    %[[PAD_CONST1:.*]] = "tosa.const"() {value = dense<{{\[\[}}0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : () -> tensor<4x2xi64>
// CHECK-DAG:    %[[PAD_CONST2:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG:    %[[PAD_ARG:.*]] = "tosa.pad"(%arg0, %[[PAD_CONST1]], %[[PAD_CONST2]]) : (tensor<5x5x32x32xf32>, tensor<4x2xi64>, tensor<f32>) -> tensor<5x5x34x34xf32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%[[PAD_ARG]], %3) : (tensor<5x5x34x34xf32>, tensor<4xi32>) -> tensor<5x34x34x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = [3, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<5x34x34x5xf32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %6) : (tensor<5x32x32x5xf32>, tensor<4xi32>) -> tensor<5x5x32x32xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x32x32xf32>

// -----

/// Test the behavior of AveragePool with non uniform padding and count_include_pad == 1
func.func @test_averagepool_pad_nonunif_with_count_include_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32> {
  %0 = "onnx.AveragePool"(%arg0) {count_include_pad = 1 : si64, ceil_mode = 0 : si64, kernel_shape = [5,3], pads = [0, 1, 2, 3] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32>
  "func.return"(%0) : (tensor<5x5x30x34xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_averagepool_pad_nonunif_with_count_include_pad(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32> {
// CHECK-DAG:    %[[PAD_CONST1:.*]] = "tosa.const"() {value = dense<{{\[\[}}0, 0], [0, 0], [0, 2], [1, 3]]> : tensor<4x2xi64>} : () -> tensor<4x2xi64>
// CHECK-DAG:    %[[PAD_CONST2:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG:    %[[PAD_ARG:.*]] = "tosa.pad"(%arg0, %[[PAD_CONST1]], %[[PAD_CONST2]]) : (tensor<5x5x32x32xf32>, tensor<4x2xi64>, tensor<f32>) -> tensor<5x5x34x36xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:    %[[TRANS_ARG:.*]] = "tosa.transpose"(%[[PAD_ARG]], %3) : (tensor<5x5x34x36xf32>, tensor<4xi32>) -> tensor<5x34x36x5xf32>
// CHECK-DAG:    %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = [5, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<5x34x36x5xf32>) -> tensor<5x30x34x5xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:    %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %6) : (tensor<5x30x34x5xf32>, tensor<4xi32>) -> tensor<5x5x30x34xf32>
// CHECK-DAG:    return %[[TRANS_MPOOL_RES]] : tensor<5x5x30x34xf32>

// -----

/// Test the behavior of AveragePool with ceiling set (Should change the result shape)
func.func @test_averagepool_strides_nonunifpad_ceil_with_count_include_pad(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x16x17xf32> {
  %0 = "onnx.AveragePool"(%arg0) {count_include_pad = 1 : si64, ceil_mode = 1 : si64, kernel_shape = [2,2], pads = [1, 2, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x16x17xf32>
  "func.return"(%0) : (tensor<5x5x16x17xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_averagepool_strides_nonunifpad_ceil_with_count_include_pad(%arg0: tensor<5x5x30x32xf32>) -> tensor<5x5x16x17xf32> {
// CHECK-DAG:    %[[PAD_CONST1:.*]] = "tosa.const"() {value = dense<{{\[\[}}0, 0], [0, 0], [1, 0], [2, 0]]> : tensor<4x2xi64>} : () -> tensor<4x2xi64>
// CHECK-DAG:    %[[PAD_CONST2:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG:    %[[PAD_ARG:.*]] = "tosa.pad"(%arg0, %[[PAD_CONST1]], %[[PAD_CONST2]]) : (tensor<5x5x30x32xf32>, tensor<4x2xi64>, tensor<f32>) -> tensor<5x5x31x34xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:    %[[TRANS_ARG:.*]] = "tosa.transpose"(%[[PAD_ARG]], %3) : (tensor<5x5x31x34xf32>, tensor<4xi32>) -> tensor<5x31x34x5xf32>
// CHECK-DAG:    %[[MPOOL_RES:.*]] = "tosa.avg_pool2d"(%[[TRANS_ARG]]) {kernel = [2, 2], pad = [0, 2, 0, 0], stride = [2, 2]} : (tensor<5x31x34x5xf32>) -> tensor<5x16x17x5xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:    %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %6) : (tensor<5x16x17x5xf32>, tensor<4xi32>) -> tensor<5x5x16x17xf32>
// CHECK-DAG:    return %[[TRANS_MPOOL_RES]] : tensor<5x5x16x17xf32>