// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// Shape inference of Pad with an `axes` input. `pads` holds one begin then one
// end value per entry of `axes`, in `axes` order; the axes need not be sorted
// or contiguous.

// -----

// Non-contiguous axes: pads = [b0, b2, e0, e2].
func.func @test_pad_axes_non_contiguous(%arg0 : tensor<2x3x4x5xf32>) -> tensor<*xf32> {
  %pads = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %axes = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3x4x5xf32>, tensor<4xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_pad_axes_non_contiguous
// CHECK:          [[RES_:%.+]] = "onnx.Pad"{{.*}} -> tensor<6x3x10x5xf32>
// CHECK:          onnx.Return [[RES_]] : tensor<6x3x10x5xf32>
}

// -----

// Scrambled axes: pads = [b2, b0, e2, e0], the same padding as above.
func.func @test_pad_axes_scrambled(%arg0 : tensor<2x3x4x5xf32>) -> tensor<*xf32> {
  %pads = onnx.Constant dense<[2, 1, 4, 3]> : tensor<4xi64>
  %axes = onnx.Constant dense<[2, 0]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3x4x5xf32>, tensor<4xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_pad_axes_scrambled
// CHECK:          [[RES_:%.+]] = "onnx.Pad"{{.*}} -> tensor<6x3x10x5xf32>
// CHECK:          onnx.Return [[RES_]] : tensor<6x3x10x5xf32>
}

// -----

// Negative, scrambled axes: [-1, -3] are axes 3 and 1.
func.func @test_pad_axes_negative(%arg0 : tensor<2x3x4x5xf32>) -> tensor<*xf32> {
  %pads = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %axes = onnx.Constant dense<[-1, -3]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3x4x5xf32>, tensor<4xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_pad_axes_negative
// CHECK:          [[RES_:%.+]] = "onnx.Pad"{{.*}} -> tensor<2x9x4x9xf32>
// CHECK:          onnx.Return [[RES_]] : tensor<2x9x4x9xf32>
}

// -----

func.func @test_pad_axes_i32(%arg0 : tensor<2x3xf32>) -> tensor<*xf32> {
  %pads = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %axes = onnx.Constant dense<[1]> : tensor<1xi32>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3xf32>, tensor<2xi64>, none, tensor<1xi32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_pad_axes_i32
// CHECK:          [[RES_:%.+]] = "onnx.Pad"{{.*}} -> tensor<2x6xf32>
// CHECK:          onnx.Return [[RES_]] : tensor<2x6xf32>
}

// -----

// Empty axes: nothing is padded.
func.func @test_pad_axes_empty(%arg0 : tensor<2x3xf32>) -> tensor<*xf32> {
  %pads = onnx.Constant dense<> : tensor<0xi64>
  %axes = onnx.Constant dense<> : tensor<0xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3xf32>, tensor<0xi64>, none, tensor<0xi64>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_pad_axes_empty
// CHECK:          [[RES_:%.+]] = "onnx.Pad"{{.*}} -> tensor<2x3xf32>
// CHECK:          onnx.Return [[RES_]] : tensor<2x3xf32>
}

// -----

// As in the ONNX Attention function body: pad the mask's last axis with
// runtime pads; the other axis keeps its size.
func.func @test_pad_axes_dynamic_pads(%arg0 : tensor<4x?xi64>, %arg1 : tensor<2xi64>, %arg2 : tensor<1xi64>) -> tensor<*xi64> {
  %axes = onnx.Constant dense<[-1]> : tensor<1xi64>
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2, %axes) {mode = "constant"} : (tensor<4x?xi64>, tensor<2xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<*xi64>
  onnx.Return %0 : tensor<*xi64>

// CHECK-LABEL:  func.func @test_pad_axes_dynamic_pads
// CHECK:          [[RES_:%.+]] = "onnx.Pad"{{.*}} -> tensor<4x?xi64>
// CHECK:          onnx.Return [[RES_]] : tensor<4x?xi64>
}

// -----

// Axes from a Concat of constants are read element by element, before
// constant propagation folds the Concat.
func.func @test_pad_axes_concat_of_constants(%arg0 : tensor<2x3x4x5xf32>) -> tensor<*xf32> {
  %pads = onnx.Constant dense<[2, 1, 4, 3]> : tensor<4xi64>
  %a2 = onnx.Constant dense<[2]> : tensor<1xi64>
  %a0 = onnx.Constant dense<[0]> : tensor<1xi64>
  %axes = "onnx.Concat"(%a2, %a0) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3x4x5xf32>, tensor<4xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_pad_axes_concat_of_constants
// CHECK:          [[RES_:%.+]] = "onnx.Pad"{{.*}} -> tensor<6x3x10x5xf32>
// CHECK:          onnx.Return [[RES_]] : tensor<6x3x10x5xf32>
}

// -----

// One axis is not known: which axes grow is not known either, but the rank is.
func.func @test_pad_axes_partly_known(%arg0 : tensor<2x3x4xf32>, %arg1 : tensor<1xi64>) -> tensor<*xf32> {
  %pads = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %a1 = onnx.Constant dense<[1]> : tensor<1xi64>
  %axes = "onnx.Concat"(%a1, %arg1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3x4xf32>, tensor<4xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_pad_axes_partly_known
// CHECK:          [[RES_:%.+]] = "onnx.Pad"{{.*}} -> tensor<?x?x?xf32>
// CHECK:          onnx.Return [[RES_]] : tensor<?x?x?xf32>
}
