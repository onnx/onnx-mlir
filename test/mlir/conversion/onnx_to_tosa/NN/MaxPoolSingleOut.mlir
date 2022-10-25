// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

/// Test the default behavior of Max Pool with no padding
func.func @test_default_maxpoolsingleout(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  "func.return"(%0) : (tensor<5x5x30x30xf32>) -> ()
}
// CHECK-LABEL: func.func @test_default_maxpoolsingleout(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
// CHECK-DAG:   "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:   %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:   %[[MPOOL_RES:.*]] = "tosa.max_pool2d"(%[[TRANS_ARG]]) {kernel = [3, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<5x32x32x5xf32>) -> tensor<5x30x30x5xf32>
// CHECK-DAG:   "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:   %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x30x30x5xf32>, tensor<4xi32>) -> tensor<5x5x30x30xf32>
// CHECK-DAG:   return %[[TRANS_MPOOL_RES]] : tensor<5x5x30x30xf32>

// -----

/// Test the behavior of Max Pool with uniform padding
func.func @test_default_maxpoolsingleout_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [3,3], pads = [1, 1, 1, 1] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
  "func.return"(%0) : (tensor<5x5x32x32xf32>) -> ()
}
// CHECK-DAG:   func.func @test_default_maxpoolsingleout_pad(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.max_pool2d"(%[[TRANS_ARG]]) {kernel = [3, 3], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<5x32x32x5xf32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x32x32x5xf32>, tensor<4xi32>) -> tensor<5x5x32x32xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x32x32xf32>

// -----

/// Test the behavior of Max Pool with non uniform padding
func.func @test_default_maxpoolsingleout_pad_nonunif(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x31x31xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {ceil_mode = 0 : si64, kernel_shape = [5,3], pads = [2, 1, 1, 0] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x31x31xf32>
  "func.return"(%0) : (tensor<5x5x31x31xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_default_maxpoolsingleout_pad_nonunif(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x31x31xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.max_pool2d"(%[[TRANS_ARG]]) {kernel = [5, 3], pad = [2, 1, 0, 1], stride = [1, 1]} : (tensor<5x32x32x5xf32>) -> tensor<5x31x31x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x31x31x5xf32>, tensor<4xi32>) -> tensor<5x5x31x31xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x31x31xf32>
// -----

/// Test the behavior of Max Pool with strides set
func.func @test_default_maxpoolsingleout_strides(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [3,3], pads = [1, 1, 1, 1], strides = [2, 2] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32>
  "func.return"(%0) : (tensor<5x5x16x16xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_default_maxpoolsingleout_strides(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.max_pool2d"(%[[TRANS_ARG]]) {kernel = [3, 3], pad = [1, 1, 1, 1], stride = [2, 2]} : (tensor<5x32x32x5xf32>) -> tensor<5x16x16x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x16x16x5xf32>, tensor<4xi32>) -> tensor<5x5x16x16xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x16x16xf32>

// -----

/// Test the behavior of Max Pool with strides and non uniform padding 
func.func @test_default_maxpoolsingleout_strides_nonunifpad(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32>
  "func.return"(%0) : (tensor<5x5x15x16xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_default_maxpoolsingleout_strides_nonunifpad(%arg0: tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x30x32xf32>, tensor<4xi32>) -> tensor<5x30x32x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.max_pool2d"(%[[TRANS_ARG]]) {kernel = [2, 2], pad = [1, 0, 0, 0], stride = [2, 2]} : (tensor<5x30x32x5xf32>) -> tensor<5x15x16x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x15x16x5xf32>, tensor<4xi32>) -> tensor<5x5x15x16xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x15x16xf32>

// -----

/// Test the behavior of Max Pool with ceiling set (Should change the result shape)
func.func @test_default_maxpoolsingleout_strides_nonunifpad_ceil(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {ceil_mode = 1 : si64, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32>
  "func.return"(%0) : (tensor<5x5x16x16xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_default_maxpoolsingleout_strides_nonunifpad_ceil(%arg0: tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x30x32xf32>, tensor<4xi32>) -> tensor<5x30x32x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.max_pool2d"(%[[TRANS_ARG]]) {kernel = [2, 2], pad = [1, 0, 0, 0], stride = [2, 2]} : (tensor<5x30x32x5xf32>) -> tensor<5x16x16x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x16x16x5xf32>, tensor<4xi32>) -> tensor<5x5x16x16xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x16x16xf32>


// -----

func.func @test_default_maxpoolsingleout_strides_high(%arg0 : tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [4,4], strides = [4, 4] } : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
  "func.return"(%0) : (tensor<5x5x4x4xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_default_maxpoolsingleout_strides_high(%arg0: tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_ARG:.*]] = "tosa.transpose"(%arg0, %0) : (tensor<5x5x16x13xf32>, tensor<4xi32>) -> tensor<5x16x13x5xf32>
// CHECK-DAG:     %[[MPOOL_RES:.*]] = "tosa.max_pool2d"(%[[TRANS_ARG]]) {kernel = [4, 4], pad = [0, 0, 0, 0], stride = [4, 4]} : (tensor<5x16x13x5xf32>) -> tensor<5x4x4x5xf32>
// CHECK-DAG:     "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:     %[[TRANS_MPOOL_RES:.*]] = "tosa.transpose"(%[[MPOOL_RES]], %3) : (tensor<5x4x4x5xf32>, tensor<4xi32>) -> tensor<5x5x4x4xf32>
// CHECK-DAG:     return %[[TRANS_MPOOL_RES]] : tensor<5x5x4x4xf32>