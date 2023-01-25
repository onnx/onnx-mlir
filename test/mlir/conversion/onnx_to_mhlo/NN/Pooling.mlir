// RUN: onnx-mlir-opt --convert-onnx-to-mhlo --canonicalize %s -split-input-file | FileCheck %s

/// Test the default behavior of Max Pool with no padding (pad are set but should be ignored)
func.func @test_default_maxpoolsingleout(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "VALID", ceil_mode = 0 : si64, kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  "func.return"(%0) : (tensor<5x5x30x30xf32>) -> ()
}
// CHECK-LABEL:           test_default_maxpoolsingleout
// CHECK:                 %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:              ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:   }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x30x30xf32>

// -----

/// Test the default behavior of Max Pool with no padding (pad are not set, default to zero)
func.func @test_default_maxpoolsingleout_defpad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  "func.return"(%0) : (tensor<5x5x30x30xf32>) -> ()
}
// CHECK-LABEL:           test_default_maxpoolsingleout_defpad
// CHECK:                 %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:              ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:   }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x30x30xf32>

// -----

/// Test the default behavior of Max Pool with uniform padding
func.func @test_default_maxpoolsingleout_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3,3], pads = [1, 1, 1, 1] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
  "func.return"(%0) : (tensor<5x5x32x32xf32>) -> ()
}
// CHECK-LABEL:            test_default_maxpoolsingleout_pad
// CHECK:                  %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:               ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                 %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                 mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x32x32xf32>

// -----

/// Test the default behavior of Max Pool with non uniform padding
func.func @test_default_maxpoolsingleout_pad_nonunif(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x31x31xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [5,3], pads = [2, 1, 1, 0] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x31x31xf32>
  "func.return"(%0) : (tensor<5x5x31x31xf32>) -> ()
}
// CHECK-LABEL:            test_default_maxpoolsingleout_pad_nonunif
// CHECK:                  %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:               ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                 %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                 mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [2, 1], [1, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 5, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x31x31xf32>

// -----

/// Test the default behavior of Max Pool with non uniform padding
func.func @test_default_maxpoolsingleout_strides(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3,3], pads = [1, 1, 1, 1], strides = [2, 2] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32>
  "func.return"(%0) : (tensor<5x5x16x16xf32>) -> ()
}
// CHECK-LABEL:            test_default_maxpoolsingleout_strides
// CHECK:                  %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:               ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                 %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                 mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x16x16xf32>

// -----

/// Test the default behavior of Max Pool with non uniform padding
func.func @test_default_maxpoolsingleout_strides_nonunifpad(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32>
  "func.return"(%0) : (tensor<5x5x15x16xf32>) -> ()
}
// CHECK-LABEL:            test_default_maxpoolsingleout_strides_nonunifpad
// CHECK:                  %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:               ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                 %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                 mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [1, 0], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<5x5x30x32xf32>, tensor<f32>) -> tensor<5x5x15x16xf32>

// -----

/// Test the default behavior of Max Pool with non uniform padding
func.func @test_default_maxpoolsingleout_strides_nonunifpad_ceil(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32>
  "func.return"(%0) : (tensor<5x5x16x16xf32>) -> ()
}
// CHECK-LABEL:            test_default_maxpoolsingleout_strides_nonunifpad_ceil
// CHECK:                  %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:               ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                 %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                 mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<5x5x30x32xf32>, tensor<f32>) -> tensor<5x5x16x16xf32>

// -----

/// Test the default behavior of Max Pool with dilatation
func.func @test_default_maxpoolsingleout_strides_dilatation(%arg0 : tensor<5x5x8x8xf32>) -> tensor<5x5x2x2xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2,2], dilations = [2, 2], strides = [3, 3] } : (tensor<5x5x8x8xf32>) -> tensor<5x5x2x2xf32>
  "func.return"(%0) : (tensor<5x5x2x2xf32>) -> ()
}
// CHECK-LABEL:            test_default_maxpoolsingleout_strides_dilatation
// CHECK:                  %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:               ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                 %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                 mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:    }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<5x5x8x8xf32>, tensor<f32>) -> tensor<5x5x2x2xf32>

// -----

/// Test the default behavior of Max Pool with dilatation
func.func @test_default_maxpoolsingleout_upper(%arg0 : tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "SAME_UPPER", ceil_mode = 0 : si64, kernel_shape = [4,4], strides = [4, 4] } : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
  "func.return"(%0) : (tensor<5x5x4x4xf32>) -> ()
}
// CHECK-LABEL:            test_default_maxpoolsingleout_upper
// CHECK:                  %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:                 ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                   %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                   mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [0, 0], [1, 2]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 4, 4]> : tensor<4xi64>, window_strides = dense<[1, 1, 4, 4]> : tensor<4xi64>} : (tensor<5x5x16x13xf32>, tensor<f32>) -> tensor<5x5x4x4xf32>

// -----

/// Test the default behavior of Max Pool with dilatation
func.func @test_default_maxpoolsingleout_lower(%arg0 : tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "SAME_LOWER", ceil_mode = 0 : si64, kernel_shape = [4,4], strides = [4, 4] } : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
  "func.return"(%0) : (tensor<5x5x4x4xf32>) -> ()
}
// CHECK-LABEL:           test_default_maxpoolsingleout_lower
// CHECK:                 %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:              ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:   }) {padding = dense<[[0, 0], [0, 0], [0, 0], [2, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 4, 4]> : tensor<4xi64>, window_strides = dense<[1, 1, 4, 4]> : tensor<4xi64>} : (tensor<5x5x16x13xf32>, tensor<f32>) -> tensor<5x5x4x4xf32>

// -----

/// Test the default behavior of Average Pool with no padding
func.func @test_averagepool_default(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  "func.return"(%0) : (tensor<5x5x30x30xf32>) -> ()
// CHECK-LABEL: test_averagepool_default
// CHECK-DAG:     [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     [[VAR_1_:%.+]] = mhlo.constant dense<1.000000e+00> : tensor<5x5x32x32xf32>
// CHECK-NEXT:    [[VAR_2_:%.+]] = "mhlo.reduce_window"(%arg0, [[VAR_0_]]) ({
// CHECK-NEXT:      ^bb0([[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>):
// CHECK-NEXT:        [[VAR_5_:%.+]] = mhlo.add [[PARAM_1_]], [[PARAM_2_]] : tensor<f32>
// CHECK-NEXT:        mhlo.return [[VAR_5_]] : tensor<f32>
// CHECK-NEXT:    }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x30x30xf32>
// CHECK-NEXT:    [[VAR_3_:%.+]] = "mhlo.reduce_window"([[VAR_1_]], [[VAR_0_]]) ({
// CHECK-NEXT:      ^bb0([[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>):
// CHECK-NEXT:        [[VAR_5_:%.+]] = mhlo.add [[PARAM_1_]], [[PARAM_2_]] : tensor<f32>
// CHECK-NEXT:        mhlo.return [[VAR_5_]] : tensor<f32>
// CHECK-NEXT:    }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x30x30xf32>
// CHECK-NEXT:    [[VAR_4_:%.+]] = mhlo.divide [[VAR_2_]], [[VAR_3_]] : tensor<5x5x30x30xf32>
// CHECK-NEXT:    return [[VAR_4_]] : tensor<5x5x30x30xf32>
}

// -----

/// Test the behavior of Average Pool with padding
func.func @test_averagepool_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [3,3], pads = [1, 1, 1, 1]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
  "func.return"(%0) : (tensor<5x5x32x32xf32>) -> ()
// CHECK-LABEL: test_averagepool_pad
// CHECK-DAG:              [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:              [[VAR_1_:%.+]] = mhlo.constant dense<1.000000e+00> : tensor<5x5x32x32xf32>
// CHECK-NEXT:             [[VAR_2_:%.+]] = "mhlo.reduce_window"(%arg0, [[VAR_0_]]) ({
// CHECK-NEXT:               ^bb0([[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>):
// CHECK-NEXT:                 [[VAR_5_:%.+]] = mhlo.add [[PARAM_1_]], [[PARAM_2_]] : tensor<f32>
// CHECK-NEXT:                 mhlo.return [[VAR_5_]] : tensor<f32>
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x32x32xf32>
// CHECK-NEXT:             [[VAR_3_:%.+]] = "mhlo.reduce_window"([[VAR_1_]], [[VAR_0_]]) ({
// CHECK-NEXT:               ^bb0([[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>):
// CHECK-NEXT:                 [[VAR_5_:%.+]] = mhlo.add [[PARAM_1_]], [[PARAM_2_]] : tensor<f32>
// CHECK-NEXT:                 mhlo.return [[VAR_5_]] : tensor<f32>
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x32x32xf32>
// CHECK-NEXT:             [[VAR_4_:%.+]] = mhlo.divide [[VAR_2_]], [[VAR_3_]] : tensor<5x5x32x32xf32>
// CHECK-NEXT:             return [[VAR_4_]] : tensor<5x5x32x32xf32>
}

// -----

/// Test the behavior of Average Pool with count_include_pad set
func.func @test_averagepool_count_include_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [3,3], count_include_pad = 1: si64} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  "func.return"(%0) : (tensor<5x5x30x30xf32>) -> ()
// CHECK-LABEL: test_averagepool_count_include_pad
// CHECK-DAG:     [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     [[VAR_1_:%.+]] = mhlo.constant dense<9.000000e+00> : tensor<5x5x30x30xf32>
// CHECK-NEXT:    [[VAR_2_:%.+]] = "mhlo.reduce_window"(%arg0, [[VAR_0_]]) ({
// CHECK-NEXT:      ^bb0([[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>):
// CHECK-NEXT:        [[VAR_4_:%.+]] = mhlo.add %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:        mhlo.return [[VAR_4_]] : tensor<f32>
// CHECK-NEXT:    }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x30x30xf32>
// CHECK-NEXT:    [[VAR_3_:%.+]] = mhlo.divide [[VAR_2_]], [[VAR_1_]] : tensor<5x5x30x30xf32>
// CHECK-NEXT:    return [[VAR_3_]] : tensor<5x5x30x30xf32>
}

// -----

/// Test the behavior of Average Pool with dynamic batch size
func.func @test_averagepool_dynamic_shape(%arg0 : tensor<?x5x32x32xf32>) -> tensor<?x5x30x30xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [3,3]} : (tensor<?x5x32x32xf32>) -> tensor<?x5x30x30xf32>
  "func.return"(%0) : (tensor<?x5x30x30xf32>) -> ()
// CHECK-LABEL: test_averagepool_dynamic_shape
// CHECK-DAG:     [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     [[VAR_1_:%.+]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:    [[VAR_2_:%.+]] = "mhlo.reduce_window"(%arg0, [[VAR_0_]]) ({
// CHECK-NEXT:      ^bb0([[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>):
// CHECK-NEXT:        [[VAR_7_:%.+]] = mhlo.add [[PARAM_1_]], [[PARAM_2_]] : tensor<f32>
// CHECK-NEXT:        mhlo.return [[VAR_7_]] : tensor<f32>
// CHECK-NEXT:    }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<?x5x32x32xf32>, tensor<f32>) -> tensor<?x5x30x30xf32>
// CHECK-NEXT:    [[VAR_3_:%.+]] = shape.shape_of %arg0 : tensor<?x5x32x32xf32> -> tensor<4xindex>
// CHECK-NEXT:    [[VAR_4_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_1_]], [[VAR_3_]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<4xindex>) -> tensor<?x5x32x32xf32>
// CHECK-NEXT:    [[VAR_5_:%.+]] = "mhlo.reduce_window"([[VAR_4_]], [[VAR_0_]]) ({
// CHECK-NEXT:      ^bb0([[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>):
// CHECK-NEXT:        [[VAR_7_:%.+]] = mhlo.add [[PARAM_1_]], [[PARAM_2_]] : tensor<f32>
// CHECK-NEXT:        mhlo.return [[VAR_7_]] : tensor<f32>
// CHECK-NEXT:    }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<?x5x32x32xf32>, tensor<f32>) -> tensor<?x5x30x30xf32>
// CHECK-NEXT:    [[VAR_6_:%.+]] = mhlo.divide [[VAR_2_]], [[VAR_5_]] : tensor<?x5x30x30xf32>
// CHECK-NEXT:    return [[VAR_6_]] : tensor<?x5x30x30xf32>
}
