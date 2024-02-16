// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// -----

/// Test the default behavior of Max Pool with no padding (pad are set but shoudl be ignored)
func.func @test_default_maxpoolsingleout(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "VALID", ceil_mode = 0 : si64, kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL: test_default_maxpoolsingleout
// CHECK: [[RES:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "VALID", ceil_mode = 0 : si64, kernel_shape = [3, 3], storage_order = 0 : si64} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
// CHECK: onnx.Return [[RES]] : tensor<5x5x30x30xf32>


// -----

/// Test the default behavior of Max Pool with no padding (pad are not set, default to zero)
func.func @test_default_maxpoolsingleout_defpad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL: test_default_maxpoolsingleout_defpad
// CHECK: [[RES:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], storage_order = 0 : si64} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
// CHECK: onnx.Return [[RES]] : tensor<5x5x30x30xf32>


// -----

/// Test the default behavior of Max Pool with uniform padding
func.func @test_default_maxpoolsingleout_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3,3], pads = [1, 1, 1, 1] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL: test_default_maxpoolsingleout_pad
// CHECK: [[RES:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], storage_order = 0 : si64} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
// CHECK: onnx.Return [[RES]] : tensor<5x5x32x32xf32>


// -----

/// Test the default behavior of Max Pool with non uniform padding
func.func @test_default_maxpoolsingleout_pad_nonunif(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [5,3], pads = [2, 1, 1, 0] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL: test_default_maxpoolsingleout_pad_nonunif
// CHECK: [[RES:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [5, 3], pads = [2, 1, 1, 0], storage_order = 0 : si64} : (tensor<5x5x32x32xf32>) -> tensor<5x5x31x31xf32>
// CHECK: onnx.Return [[RES]] : tensor<5x5x31x31xf32>

// -----

/// Test the default behavior of Max Pool with non uniform padding
func.func @test_default_maxpoolsingleout_strides(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3,3], pads = [1, 1, 1, 1], strides = [2, 2] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL: test_default_maxpoolsingleout_strides
// CHECK: [[RES:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], storage_order = 0 : si64, strides = [2, 2]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32>
// CHECK: onnx.Return [[RES]] : tensor<5x5x16x16xf32>

// -----

/// Test the default behavior of Max Pool with non uniform padding
func.func @test_default_maxpoolsingleout_strides_nonunifpad(%arg0 : tensor<5x5x30x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL: test_default_maxpoolsingleout_strides_nonunifpad
// CHECK: [[RES:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 0], storage_order = 0 : si64, strides = [2, 2]} : (tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32>
// CHECK: onnx.Return [[RES]] : tensor<5x5x15x16xf32>

// -----

/// Test the default behavior of Max Pool with non uniform padding
func.func @test_default_maxpoolsingleout_strides_nonunifpad_ceil(%arg0 : tensor<5x5x30x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL: test_default_maxpoolsingleout_strides_nonunifpad_ceil
// CHECK: [[RES:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 0], storage_order = 0 : si64, strides = [2, 2]} : (tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32>
// CHECK: onnx.Return [[RES]] : tensor<5x5x16x16xf32>


// -----

/// Test the default behavior of Max Pool with dilatation
func.func @test_default_maxpoolsingleout_strides_dilatation(%arg0 : tensor<5x5x8x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2,2], dilations = [2, 2], strides = [3, 3] } : (tensor<5x5x8x8xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL: test_default_maxpoolsingleout_strides_dilatation
// CHECK: [[RES:%.+]] = "onnx.MaxPoolSingleOut"(%arg0)  {auto_pad = "NOTSET", ceil_mode = 0 : si64, dilations = [2, 2], kernel_shape = [2, 2], storage_order = 0 : si64, strides = [3, 3]} : (tensor<5x5x8x8xf32>) -> tensor<5x5x2x2xf32>
// CHECK: onnx.Return [[RES]] : tensor<5x5x2x2xf32>

// -----

/// Test the default behavior of Max Pool with dilatation
func.func @test_default_maxpoolsingleout_upper(%arg0 : tensor<5x5x16x13xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "SAME_UPPER", ceil_mode = 0 : si64, kernel_shape = [4,4], strides = [4, 4] } : (tensor<5x5x16x13xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL: test_default_maxpoolsingleout_upper
// CHECK: [[RES:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "SAME_UPPER", ceil_mode = 0 : si64, kernel_shape = [4, 4], storage_order = 0 : si64, strides = [4, 4]} : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
// CHECK: onnx.Return [[RES]] : tensor<5x5x4x4xf32>


// -----

/// Test the default behavior of Max Pool with dilatation
func.func @test_default_maxpoolsingleout_lower(%arg0 : tensor<5x5x16x13xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "SAME_LOWER", ceil_mode = 0 : si64, kernel_shape = [4,4], strides = [4, 4] } : (tensor<5x5x16x13xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}
// CHECK-LABEL: test_default_maxpoolsingleout_lower
// CHECK: [[RES:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "SAME_LOWER", ceil_mode = 0 : si64, kernel_shape = [4, 4], storage_order = 0 : si64, strides = [4, 4]} : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
// CHECK: onnx.Return [[RES]] : tensor<5x5x4x4xf32>

