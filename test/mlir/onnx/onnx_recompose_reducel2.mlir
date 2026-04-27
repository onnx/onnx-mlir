// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s -split-input-file | FileCheck %s

// -----

// recompose Mul(x,x) -> ReduceSum -> Sqrt to ReduceL2

func.func @test_recompose_reducel2_basic(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32> {
  %0 = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %1 = "onnx.ReduceSum"(%0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %2 = "onnx.Sqrt"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %2 : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[PARAM_1_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
// CHECK-NEXT:      onnx.Return [[VAR_0_]] : tensor<?x?xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_recompose_reducel2_keepdims(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %1 = "onnx.ReduceSum"(%0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?x?xf32>
  %2 = "onnx.Sqrt"(%1) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  onnx.Return %2 : tensor<?x?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_keepdims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?x?xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[PARAM_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?x?xf32>
// CHECK-NEXT:      onnx.Return [[VAR_0_]] : tensor<?x?x?xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_recompose_reducel2_noop_with_empty_axes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %1 = "onnx.ReduceSum"(%0, %none) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  %2 = "onnx.Sqrt"(%1) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  onnx.Return %2 : tensor<1x1x1xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_noop_with_empty_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<1x1x1xf32>
// CHECK:         }
}

// -----

func.func @test_recompose_reducel2_from_reducesumsquare(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32> {
  %0 = "onnx.ReduceSumSquare"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %1 : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_from_reducesumsquare
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[PARAM_1_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
// CHECK-NEXT:      onnx.Return [[VAR_0_]] : tensor<?x?xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_recompose_reducesumsquare(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32> {
  %0 = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %1 = "onnx.ReduceSum"(%0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  onnx.Return %1 : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducesumsquare
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "onnx.ReduceSumSquare"([[PARAM_0_]], [[PARAM_1_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
// CHECK-NEXT:      onnx.Return [[VAR_0_]] : tensor<?x?xf32>
// CHECK-NEXT:    }
}

// -----

// Mul(x,y) should break the pattern
func.func @test_recompose_reducel2_not_square(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>, %arg2: tensor<1xi64>) -> tensor<?x?xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %1 = "onnx.ReduceSum"(%0, %arg2) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %2 = "onnx.Sqrt"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %2 : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_not_square
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           "onnx.Mul"
// CHECK:           "onnx.ReduceSum"
// CHECK:           "onnx.Sqrt"
}

// -----

func.func @test_recompose_reducel2_mul_extra_use(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> (tensor<?x?xf32>, tensor<2x3x4xf32>) {
  %0 = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %1 = "onnx.ReduceSum"(%0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %2 = "onnx.Sqrt"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %2, %0 : tensor<?x?xf32>, tensor<2x3x4xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_mul_extra_use
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           "onnx.Mul"
// CHECK:           "onnx.ReduceSum"
// CHECK:           "onnx.Sqrt"
}

// -----

func.func @test_recompose_reducel2_reducesumsquare_extra_use(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = "onnx.ReduceSumSquare"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %1, %0 : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_reducesumsquare_extra_use
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK:           "onnx.ReduceSumSquare"
// CHECK:           "onnx.Sqrt"
}
