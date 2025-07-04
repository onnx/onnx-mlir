// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

/// Test the default behavior of AveragePool with no padding
func.func @test_default_averagepool(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  "func.return"(%0) : (tensor<5x5x30x30xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_default_averagepool
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x32x32xf32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.avg_pool2d [[VAR_0_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x32x32x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x30x30x5xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x30x30x5xf32>) -> tensor<5x5x30x30xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x5x30x30xf32>
// CHECK:         }

// -----

/// Test the behavior of AveragePool with uniform padding
func.func @test_default_averagepool_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [3,3], pads = [1, 1, 1, 1] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
  "func.return"(%0) : (tensor<5x5x32x32xf32>) -> ()
// CHECK-LABEL:  func.func @test_default_averagepool_pad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x32x32xf32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.avg_pool2d [[VAR_0_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<5x32x32x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x32x32x5xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x32x32x5xf32>) -> tensor<5x5x32x32xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x5x32x32xf32>
// CHECK:         }
}

// -----

/// Test the behavior of AveragePool with non uniform padding
func.func @test_default_averagepool_pad_nonunif(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x33xf32> {
  %0 = "onnx.AveragePool"(%arg0) {ceil_mode = 0 : si64, kernel_shape = [5,3], pads = [0, 1, 2,2] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x33xf32>
  "func.return"(%0) : (tensor<5x5x30x33xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_default_averagepool_pad_nonunif
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x32x32xf32>) -> tensor<5x5x30x33xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x32x32xf32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.avg_pool2d [[VAR_0_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 5, 3>, pad = array<i64: 0, 2, 1, 2>, stride = array<i64: 1, 1>} : (tensor<5x32x32x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x30x33x5xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x30x33x5xf32>) -> tensor<5x5x30x33xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x5x30x33xf32>
// CHECK:         }

// -----

/// Test the behavior of AveragePool with strides set
func.func @test_default_averagepool_strides(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [4,4], pads = [1, 1, 1, 1], strides = [2, 2] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32>
  "func.return"(%0) : (tensor<5x5x16x16xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_default_averagepool_strides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x32x32xf32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.avg_pool2d [[VAR_0_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 4, 4>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<5x32x32x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x16x16x5xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x16x16x5xf32>) -> tensor<5x5x16x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x5x16x16xf32>
// CHECK:         }

// -----

/// Test the behavior of AveragePool with strides and non uniform padding 
func.func @test_default_averagepool_strides_nonunifpad(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [3,3], pads = [1, 1, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32>
  "func.return"(%0) : (tensor<5x5x15x16xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_default_averagepool_strides_nonunifpad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x30x32xf32>) -> tensor<5x30x32x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.avg_pool2d [[VAR_0_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>} : (tensor<5x30x32x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x15x16x5xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x15x16x5xf32>) -> tensor<5x5x15x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x5x15x16xf32>
// CHECK:         }

// -----

/// Test the behavior of AveragePool with ceiling set (Should change the result shape)
func.func @test_default_averagepool_strides_nonunifpad_ceil(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
  %0 = "onnx.AveragePool"(%arg0) {ceil_mode = 1 : si64, kernel_shape = [3,3], pads = [1, 1, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32>
  "func.return"(%0) : (tensor<5x5x15x16xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_default_averagepool_strides_nonunifpad_ceil
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x30x32xf32>) -> tensor<5x30x32x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.avg_pool2d [[VAR_0_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>} : (tensor<5x30x32x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x15x16x5xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x15x16x5xf32>) -> tensor<5x5x15x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x5x15x16xf32>
// CHECK:         }


// -----

func.func @test_default_averagepool_autopad_valid(%arg0 : tensor<5x5x16x13xf32>) -> tensor<5x5x14x11xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "VALID", kernel_shape = [3,3]} : (tensor<5x5x16x13xf32>) -> tensor<5x5x14x11xf32>
  "func.return"(%0) : (tensor<5x5x14x11xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_default_averagepool_autopad_valid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x16x13xf32>) -> tensor<5x5x14x11xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x16x13xf32>) -> tensor<5x16x13x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.avg_pool2d [[VAR_0_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x16x13x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x14x11x5xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x14x11x5xf32>) -> tensor<5x5x14x11xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x5x14x11xf32>
// CHECK:         }

// -----

func.func @test_default_averagepool_same_upper_ceil_mode(%arg0 : tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "SAME_UPPER", ceil_mode = 1 : si64, kernel_shape = [4,4], strides = [4, 4] } : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
  "func.return"(%0) : (tensor<5x5x4x4xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_default_averagepool_same_upper_ceil_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x16x13xf32>) -> tensor<5x16x13x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.avg_pool2d [[VAR_0_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 4, 4>, pad = array<i64: 0, 0, 1, 2>, stride = array<i64: 4, 4>} : (tensor<5x16x13x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x4x4x5xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x4x4x5xf32>) -> tensor<5x5x4x4xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x5x4x4xf32>
// CHECK:         }

// -----

/// Test the behavior of AveragePool with uniform padding and count_include_pad == 1
func.func @test_averagepool_pad_with_count_include_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {count_include_pad = 1 : si64, kernel_shape = [3,3], pads = [1, 0, 1, 0] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_averagepool_pad_with_count_include_pad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x32x32xf32>) -> tensor<5x5x32x30xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {values = dense<[0, 0, 0, 0, 1, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<5x5x32x32xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<5x5x34x32xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x34x32xf32>) -> tensor<5x34x32x5xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.avg_pool2d [[VAR_3_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x34x32x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x32x30x5xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x32x30x5xf32>) -> tensor<5x5x32x30xf32>
// CHECK:           return [[VAR_5_]] : tensor<5x5x32x30xf32>
// CHECK:         }

// -----

/// Test the behavior of AveragePool with non uniform padding and count_include_pad == 1
func.func @test_averagepool_pad_nonunif_with_count_include_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32> {
  %0 = "onnx.AveragePool"(%arg0) {count_include_pad = 1 : si64, ceil_mode = 0 : si64, kernel_shape = [5,3], pads = [0, 1, 2, 3] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32>
  "func.return"(%0) : (tensor<5x5x30x34xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_averagepool_pad_nonunif_with_count_include_pad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {values = dense<[0, 0, 0, 0, 0, 2, 1, 3]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<5x5x32x32xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<5x5x34x36xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x34x36xf32>) -> tensor<5x34x36x5xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.avg_pool2d [[VAR_3_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 5, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x34x36x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x30x34x5xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x30x34x5xf32>) -> tensor<5x5x30x34xf32>
// CHECK:           return [[VAR_5_]] : tensor<5x5x30x34xf32>
// CHECK:         }

// -----

/// Test the behavior of AveragePool with ceiling set (Should change the result shape)
func.func @test_averagepool_strides_nonunifpad_ceil_with_count_include_pad(%arg0 : tensor<5x5x30x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {count_include_pad = 1 : si64, ceil_mode = 1 : si64, kernel_shape = [3,2], pads = [1, 2, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL:  func.func @test_averagepool_strides_nonunifpad_ceil_with_count_include_pad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x30x32xf32>) -> tensor<5x5x15x17xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {values = dense<[0, 0, 0, 0, 1, 0, 2, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<5x5x30x32xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<5x5x31x34xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_2_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x5x31x34xf32>) -> tensor<5x31x34x5xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.avg_pool2d [[VAR_3_]], [[VAR_1_]], [[VAR_1_]] {acc_type = f32, kernel = array<i64: 3, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<5x31x34x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x15x17x5xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x15x17x5xf32>) -> tensor<5x5x15x17xf32>
// CHECK:           return [[VAR_5_]] : tensor<5x5x15x17xf32>
// CHECK:         }
