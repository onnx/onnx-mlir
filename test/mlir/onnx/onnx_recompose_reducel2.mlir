// RUN: onnx-mlir-opt --recompose-onnx=enable-reducel2-recompositions=true --canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s -split-input-file | FileCheck %s --check-prefix=DISABLED-CHECK

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

// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_basic
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"
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

// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_keepdims
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"

// -----

func.func @test_recompose_reducel2_noop_with_empty_axes(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %1 = "onnx.ReduceSum"(%0, %none) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<2x3x4xf32>
  %2 = "onnx.Sqrt"(%1) : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  onnx.Return %2 : tensor<2x3x4xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_noop_with_empty_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<2x3x4xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<2x3x4xf32>
// CHECK:         }
}

// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_noop_with_empty_axes
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"

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

// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_from_reducesumsquare
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"

// -----

// An incomplete chain (no Sqrt/Pow consumer) must stay decomposed: fusing it
// into ReduceSumSquare would be undone by the unconditional ReduceSumSquare
// decomposition and the two would never converge.
func.func @test_mul_reducesum_stays_decomposed(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32> {
  %0 = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %1 = "onnx.ReduceSum"(%0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  onnx.Return %1 : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_mul_reducesum_stays_decomposed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?xf32> {
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           [[SQUARE_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]])
// CHECK:           [[SUM_:%.+]] = "onnx.ReduceSum"([[SQUARE_]], [[PARAM_1_]])
// CHECK:           onnx.Return [[SUM_]] : tensor<?x?xf32>
// CHECK-NEXT:    }
}

// DISABLED-CHECK-LABEL:  func.func @test_mul_reducesum_stays_decomposed
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"

// -----

func.func @test_pow_reducesum_stays_decomposed(%arg0: tensor<1x512x10xf32>, %arg1: tensor<1xi64>) -> tensor<1x1x10xf32> {
  %0 = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %1 = "onnx.Pow"(%arg0, %0) : (tensor<1x512x10xf32>, tensor<f32>) -> tensor<1x512x10xf32>
  %2 = "onnx.ReduceSum"(%1, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x512x10xf32>, tensor<1xi64>) -> tensor<1x1x10xf32>
  onnx.Return %2 : tensor<1x1x10xf32>
// CHECK-LABEL:  func.func @test_pow_reducesum_stays_decomposed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x512x10xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<1x1x10xf32> {
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           [[SQUARE_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]])
// CHECK:           [[SUM_:%.+]] = "onnx.ReduceSum"([[SQUARE_]], [[PARAM_1_]])
// CHECK:           onnx.Return [[SUM_]] : tensor<1x1x10xf32>
// CHECK:         }
}

// DISABLED-CHECK-LABEL:  func.func @test_pow_reducesum_stays_decomposed
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"

// -----

func.func @test_recompose_reducel2_from_pow_reduce_sum_pow(%arg0: tensor<1x512x10xf32>, %arg1: tensor<1xi64>) -> tensor<1x1x10xf32> {
  %0 = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %1 = "onnx.Pow"(%arg0, %0) : (tensor<1x512x10xf32>, tensor<f32>) -> tensor<1x512x10xf32>
  %2 = "onnx.ReduceSum"(%1, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x512x10xf32>, tensor<1xi64>) -> tensor<1x1x10xf32>
  %3 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %4 = "onnx.Pow"(%2, %3) : (tensor<1x1x10xf32>, tensor<f32>) -> tensor<1x1x10xf32>
  onnx.Return %4 : tensor<1x1x10xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_from_pow_reduce_sum_pow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x512x10xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<1x1x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[PARAM_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x512x10xf32>, tensor<1xi64>) -> tensor<1x1x10xf32>
// CHECK-NEXT:      onnx.Return [[VAR_0_]] : tensor<1x1x10xf32>
// CHECK-NEXT:    }
}

// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_from_pow_reduce_sum_pow
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"

// -----

// recompose Pow(x,2) -> ReduceSum -> Sqrt to ReduceL2
func.func @test_recompose_reducel2_from_sqrt_reduce_sum_pow(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32> {
  %two = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %square = "onnx.Pow"(%arg0, %two) : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x3x4xf32>
  %sum = "onnx.ReduceSum"(%square, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %result = "onnx.Sqrt"(%sum) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %result : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_from_sqrt_reduce_sum_pow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?xf32> {
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           [[L2_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[PARAM_1_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
// CHECK:           onnx.Return [[L2_]] : tensor<?x?xf32>
// CHECK:         }
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_from_sqrt_reduce_sum_pow
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"
}

// -----

// recompose Mul(x,x) -> ReduceSum -> Pow(y,0.5) to ReduceL2
func.func @test_recompose_reducel2_from_pow_reduce_sum_mul(%arg0: tensor<2x3x4xf32>) -> tensor<f32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %square = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %sum = "onnx.ReduceSum"(%square, %none) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, none) -> tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %result = "onnx.Pow"(%sum, %half) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  onnx.Return %result : tensor<f32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_from_pow_reduce_sum_mul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<f32> {
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           [[NONE_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[L2_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[NONE_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
// CHECK:           onnx.Return [[L2_]] : tensor<f32>
// CHECK:         }
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_from_pow_reduce_sum_mul
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"
}

// -----

// recompose ReduceSumSquare -> Pow(y,0.5) to ReduceL2, including explicit
// empty axes and noop_with_empty_axes.
func.func @test_recompose_reducel2_from_pow_reducesumsquare_empty_axes(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %axes = onnx.Constant dense<> : tensor<0xi64>
  %sum_square = "onnx.ReduceSumSquare"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, tensor<0xi64>) -> tensor<2x3x4xf32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %result = "onnx.Pow"(%sum_square, %half) : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x3x4xf32>
  onnx.Return %result : tensor<2x3x4xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_from_pow_reducesumsquare_empty_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
// CHECK:           [[AXES_:%.+]] = onnx.Constant dense<> : tensor<0xi64>
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           [[L2_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[AXES_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64}
// CHECK:           onnx.Return [[L2_]] : tensor<2x3x4xf32>
// CHECK:         }
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_from_pow_reducesumsquare_empty_axes
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
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

// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_not_square
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"

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

// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_mul_extra_use
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"

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

// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_reducesumsquare_extra_use
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"

// -----

func.func @test_recompose_reducel2_reducesum_extra_use(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %square = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %sum = "onnx.ReduceSum"(%square, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %result = "onnx.Sqrt"(%sum) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %result, %sum : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_reducesum_extra_use
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           "onnx.Mul"
// CHECK:           [[SUM_:%.+]] = "onnx.ReduceSum"
// CHECK:           "onnx.Sqrt"([[SUM_]])
// CHECK:           onnx.Return {{%.+}}, [[SUM_]]
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_reducesum_extra_use
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"
}

// -----

func.func @test_recompose_reducel2_pow_square_extra_use(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> (tensor<?x?xf32>, tensor<2x3x4xf32>) {
  %two = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %square = "onnx.Pow"(%arg0, %two) : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x3x4xf32>
  %sum = "onnx.ReduceSum"(%square, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %result = "onnx.Sqrt"(%sum) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %result, %square : tensor<?x?xf32>, tensor<2x3x4xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_pow_square_extra_use
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           [[SQUARE_:%.+]] = "onnx.Mul"
// CHECK:           "onnx.ReduceSum"([[SQUARE_]]
// CHECK:           "onnx.Sqrt"
// CHECK:           onnx.Return {{%.+}}, [[SQUARE_]]
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_pow_square_extra_use
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"
}

// -----

func.func @test_recompose_reducel2_wrong_inner_exponent(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32> {
  %three = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %power = "onnx.Pow"(%arg0, %three) : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x3x4xf32>
  %sum = "onnx.ReduceSum"(%power, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %result = "onnx.Sqrt"(%sum) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %result : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_wrong_inner_exponent
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           "onnx.Pow"
// CHECK:           "onnx.ReduceSum"
// CHECK:           "onnx.Sqrt"
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_wrong_inner_exponent
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"
}

// -----

func.func @test_recompose_reducel2_nonconstant_inner_exponent(%arg0: tensor<2x3x4xf32>, %arg1: tensor<f32>, %arg2: tensor<1xi64>) -> tensor<?x?xf32> {
  %power = "onnx.Pow"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x3x4xf32>
  %sum = "onnx.ReduceSum"(%power, %arg2) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %result = "onnx.Sqrt"(%sum) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %result : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_nonconstant_inner_exponent
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           "onnx.Pow"
// CHECK:           "onnx.ReduceSum"
// CHECK:           "onnx.Sqrt"
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_nonconstant_inner_exponent
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"
}

// -----

func.func @test_recompose_reducel2_wrong_outer_exponent(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32> {
  %square = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %sum = "onnx.ReduceSum"(%square, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %result = "onnx.Pow"(%sum, %one) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  onnx.Return %result : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_wrong_outer_exponent
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           "onnx.Mul"
// CHECK:           [[SUM_:%.+]] = "onnx.ReduceSum"
// CHECK:           onnx.Return [[SUM_]] : tensor<?x?xf32>
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_wrong_outer_exponent
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"
}

// -----

func.func @test_recompose_reducel2_nonconstant_outer_exponent(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>, %arg2: tensor<f32>) -> tensor<?x?xf32> {
  %square = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %sum = "onnx.ReduceSum"(%square, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %result = "onnx.Pow"(%sum, %arg2) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  onnx.Return %result : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_nonconstant_outer_exponent
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK-NOT:       "onnx.ReduceSumSquare"
// CHECK:           "onnx.Mul"
// CHECK:           "onnx.ReduceSum"
// CHECK:           "onnx.Pow"
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_nonconstant_outer_exponent
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
// DISABLED-CHECK-NOT:       "onnx.ReduceSumSquare"
}

// -----

// Uniform non-scalar exponents are safe when broadcasting does not expand
// either Pow input.
func.func @test_recompose_reducel2_non_scalar_shape_preserving_exponents(%arg0: tensor<2x3xf32>) -> tensor<3xf32> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %two = onnx.Constant dense<2.000000e+00> : tensor<2x1xf32>
  %square = "onnx.Pow"(%arg0, %two) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  %sum = "onnx.ReduceSum"(%square, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<3xf32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<3xf32>
  %result = "onnx.Pow"(%sum, %half) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  onnx.Return %result : tensor<3xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_non_scalar_shape_preserving_exponents
// CHECK:           [[L2_:%.+]] = "onnx.ReduceL2"
// CHECK:           onnx.Return [[L2_]] : tensor<3xf32>
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_non_scalar_shape_preserving_exponents
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
}

// -----

// A non-scalar exponent can broadcast and expand the input, so this is not
// equivalent to ReduceL2(%arg0, %axes).
func.func @test_recompose_reducel2_broadcast_inner_exponent(%arg0: tensor<1x3xf32>) -> tensor<3xf32> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %two = onnx.Constant dense<2.000000e+00> : tensor<2x1xf32>
  %square = "onnx.Pow"(%arg0, %two) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  %sum = "onnx.ReduceSum"(%square, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<3xf32>
  %result = "onnx.Sqrt"(%sum) : (tensor<3xf32>) -> tensor<3xf32>
  onnx.Return %result : tensor<3xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_broadcast_inner_exponent
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK:           "onnx.Sqrt"
// CHECK:           onnx.Return
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_broadcast_inner_exponent
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
}

// -----

// The outer exponent can also broadcast the reduced tensor and must not be
// dropped by replacing Pow with ReduceL2.
func.func @test_recompose_reducel2_broadcast_outer_exponent(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %square = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %sum = "onnx.ReduceSum"(%square, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<3xf32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<2x1xf32>
  %result = "onnx.Pow"(%sum, %half) : (tensor<3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  onnx.Return %result : tensor<2x3xf32>
// CHECK-LABEL:  func.func @test_recompose_reducel2_broadcast_outer_exponent
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK:           "onnx.Pow"
// CHECK:           onnx.Return
// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_broadcast_outer_exponent
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"
}

// -----

func.func @test_recompose_reducel2_from_mul_reducesum_quant_types(%arg0: tensor<2x3x4x!quant.uniform<i8:f32, 0.5:0>>, %arg1: tensor<1xi64>) -> tensor<?x?x!quant.uniform<i8:f32, 4.0:3>> {
  %0 = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<2x3x4x!quant.uniform<i8:f32, 0.5:0>>) -> tensor<2x3x4x!quant.uniform<i8:f32, 1.0:1>>
  %1 = "onnx.ReduceSum"(%0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4x!quant.uniform<i8:f32, 1.0:1>>, tensor<1xi64>) -> tensor<?x?x!quant.uniform<i8:f32, 2.0:2>>
  %2 = "onnx.Sqrt"(%1) : (tensor<?x?x!quant.uniform<i8:f32, 2.0:2>>) -> tensor<?x?x!quant.uniform<i8:f32, 4.0:3>>
  onnx.Return %2 : tensor<?x?x!quant.uniform<i8:f32, 4.0:3>>
// CHECK-LABEL:  func.func @test_recompose_reducel2_from_mul_reducesum_quant_types
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK:           "onnx.Mul"
// CHECK:           "onnx.ReduceSum"
// CHECK:           "onnx.Sqrt"
}

// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_from_mul_reducesum_quant_types
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"

// -----

func.func @test_recompose_reducel2_quant_types(%arg0: tensor<2x3x4x!quant.uniform<i8:f32, 0.5:0>>, %arg1: tensor<1xi64>) -> tensor<?x?x!quant.uniform<i8:f32, 2.0:2>> {
  %0 = "onnx.ReduceSumSquare"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4x!quant.uniform<i8:f32, 0.5:0>>, tensor<1xi64>) -> tensor<?x?x!quant.uniform<i8:f32, 1.0:1>>
  %1 = "onnx.Sqrt"(%0) : (tensor<?x?x!quant.uniform<i8:f32, 1.0:1>>) -> tensor<?x?x!quant.uniform<i8:f32, 2.0:2>>
  onnx.Return %1 : tensor<?x?x!quant.uniform<i8:f32, 2.0:2>>
// CHECK-LABEL:  func.func @test_recompose_reducel2_quant_types
// CHECK-NOT:       "onnx.ReduceL2"
// CHECK:           "onnx.ReduceSumSquare"
// CHECK:           "onnx.Sqrt"
}

// DISABLED-CHECK-LABEL:  func.func @test_recompose_reducel2_quant_types
// DISABLED-CHECK-NOT:       "onnx.ReduceL2"