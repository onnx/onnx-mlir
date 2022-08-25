// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s -split-input-file | FileCheck %s

/// Test the default behavior of Max Pool with no padding (pad are set but shoudl be ignored)
func.func @test_default_maxpoolsingleout(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "VALID", ceil_mode = 0 : si64, kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  "func.return"(%0) : (tensor<5x5x30x30xf32>) -> ()
}
// CHECK-LABEL:           test_default_maxpoolsingleout
// CHECK:                 %1 = "mhlo.reduce_window"(%arg0, %0) ({
// CHECK-NEXT:              ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:                %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:                mhlo.return %2 : tensor<f32>
// CHECK-NEXT{LITERAL}:   }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : vector<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : vector<4xi64>, window_strides = dense<1> : vector<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x30x30xf32>


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
// CHECK-NEXT{LITERAL}:   }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : vector<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : vector<4xi64>, window_strides = dense<1> : vector<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x30x30xf32>


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
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : vector<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : vector<4xi64>, window_strides = dense<1> : vector<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x32x32xf32>


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
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [2, 1], [1, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : vector<4xi64>, window_dimensions = dense<[1, 1, 5, 3]> : vector<4xi64>, window_strides = dense<1> : vector<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x31x31xf32>

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
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : vector<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : vector<4xi64>, window_strides = dense<[1, 1, 2, 2]> : vector<4xi64>} : (tensor<5x5x32x32xf32>, tensor<f32>) -> tensor<5x5x16x16xf32>

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
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [1, 0], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : vector<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : vector<4xi64>, window_strides = dense<[1, 1, 2, 2]> : vector<4xi64>} : (tensor<5x5x30x32xf32>, tensor<f32>) -> tensor<5x5x15x16xf32>

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
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : vector<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : vector<4xi64>, window_strides = dense<[1, 1, 2, 2]> : vector<4xi64>} : (tensor<5x5x30x32xf32>, tensor<f32>) -> tensor<5x5x16x16xf32>


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
// CHECK-NEXT{LITERAL}:    }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<[1, 1, 2, 2]> : vector<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : vector<4xi64>, window_strides = dense<[1, 1, 3, 3]> : vector<4xi64>} : (tensor<5x5x8x8xf32>, tensor<f32>) -> tensor<5x5x2x2xf32>

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
// CHECK-NEXT{LITERAL}:    }) {padding = dense<[[0, 0], [0, 0], [0, 0], [1, 2]]> : tensor<4x2xi64>, window_dilations = dense<1> : vector<4xi64>, window_dimensions = dense<[1, 1, 4, 4]> : vector<4xi64>, window_strides = dense<[1, 1, 4, 4]> : vector<4xi64>} : (tensor<5x5x16x13xf32>, tensor<f32>) -> tensor<5x5x4x4xf32>


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
// CHECK-NEXT{LITERAL}:   }) {padding = dense<[[0, 0], [0, 0], [0, 0], [2, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : vector<4xi64>, window_dimensions = dense<[1, 1, 4, 4]> : vector<4xi64>, window_strides = dense<[1, 1, 4, 4]> : vector<4xi64>} : (tensor<5x5x16x13xf32>, tensor<f32>) -> tensor<5x5x4x4xf32>
