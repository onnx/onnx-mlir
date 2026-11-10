// Baseline for the Where(Not(c), x, y) tests in onnx_canonicalization.mlir, used
// by utils/GroundLitTest.py in file mode. Each function computes Not(c) as
// Xor(c, true), which no canonicalization turns back into Not, so the
// WhereNotConditionSwapPattern cannot fire here.

// -----

func.func @test_where_not_cond_swap(%arg0: tensor<1x1x?x?xi1> {onnx.name = "c"}, %arg1: tensor<f32> {onnx.name = "x"}, %arg2: tensor<?x1x?x?xf32> {onnx.name = "y"}) -> (tensor<?x1x?x?xf32> {onnx.name = "z"}) {
  %true = onnx.Constant dense<true> : tensor<i1>
  %0 = "onnx.Xor"(%arg0, %true) : (tensor<1x1x?x?xi1>, tensor<i1>) -> tensor<1x1x?x?xi1>
  %1 = "onnx.Where"(%0, %arg1, %arg2) : (tensor<1x1x?x?xi1>, tensor<f32>, tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  onnx.Return %1 : tensor<?x1x?x?xf32>
}
// CHECK-LABEL: func.func @test_where_not_cond_swap

// -----

func.func @test_where_not_cond_swap_multi_use(%arg0: tensor<?x?xi1> {onnx.name = "c"}, %arg1: tensor<?x?xf32> {onnx.name = "x"}, %arg2: tensor<?x?xf32> {onnx.name = "y"}) -> (tensor<?x?xf32> {onnx.name = "z"}, tensor<?x?xi1> {onnx.name = "w"}) {
  %true = onnx.Constant dense<true> : tensor<i1>
  %0 = "onnx.Xor"(%arg0, %true) : (tensor<?x?xi1>, tensor<i1>) -> tensor<?x?xi1>
  %1 = "onnx.Where"(%0, %arg1, %arg2) : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "onnx.And"(%0, %arg0) : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
  onnx.Return %1, %2 : tensor<?x?xf32>, tensor<?x?xi1>
}
// CHECK-LABEL: func.func @test_where_not_cond_swap_multi_use

// -----

func.func @test_where_not_cond_swap_two_wheres(%arg0: tensor<3x5xi1> {onnx.name = "c"}, %arg1: tensor<3x5xf32> {onnx.name = "x"}, %arg2: tensor<3x5xf32> {onnx.name = "y"}) -> (tensor<3x5xf32> {onnx.name = "z"}, tensor<3x5xf32> {onnx.name = "w"}) {
  %true = onnx.Constant dense<true> : tensor<i1>
  %0 = "onnx.Xor"(%arg0, %true) : (tensor<3x5xi1>, tensor<i1>) -> tensor<3x5xi1>
  %1 = "onnx.Where"(%0, %arg1, %arg2) : (tensor<3x5xi1>, tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  %2 = "onnx.Where"(%0, %arg2, %arg1) : (tensor<3x5xi1>, tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  onnx.Return %1, %2 : tensor<3x5xf32>, tensor<3x5xf32>
}
// CHECK-LABEL: func.func @test_where_not_cond_swap_two_wheres

// -----

func.func @test_where_not_cond_swap_through_unsqueeze(%arg0: tensor<?x?xi1> {onnx.name = "c"}, %arg1: tensor<1x?x?xf32> {onnx.name = "x"}, %arg2: tensor<1x?x?xf32> {onnx.name = "y"}) -> (tensor<1x?x?xf32> {onnx.name = "z"}) {
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %true = onnx.Constant dense<true> : tensor<i1>
  %0 = "onnx.Xor"(%arg0, %true) : (tensor<?x?xi1>, tensor<i1>) -> tensor<?x?xi1>
  %1 = "onnx.Unsqueeze"(%0, %axes) : (tensor<?x?xi1>, tensor<1xi64>) -> tensor<1x?x?xi1>
  %2 = "onnx.Where"(%1, %arg1, %arg2) : (tensor<1x?x?xi1>, tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
  onnx.Return %2 : tensor<1x?x?xf32>
}
// CHECK-LABEL: func.func @test_where_not_cond_swap_through_unsqueeze

// -----

// Baselines for the SinkExpandBelow* tests in onnx_canonicalization.mlir. For
// the positive cases, Expand(x, s) is written as Add(x, Expand(0, s)): the
// sink patterns decline an Expand of a constant, and
// PropagateScalarConstantExpandPattern needs both Add operands constant, so
// the computation stays at full size. Negative cases use the equivalent
// sunk or broadcast form instead.

func.func @test_sink_expand_below_cast(%arg0: tensor<?x1x7xi64> {onnx.name = "x"}) -> (tensor<?x5x7xf32> {onnx.name = "y"}) {
  %zero = onnx.Constant dense<0> : tensor<i64>
  %c57 = onnx.Constant dense<[5, 7]> : tensor<2xi64>
  %d0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x1x7xi64>) -> tensor<1xi64>
  %s = "onnx.Concat"(%d0, %c57) {axis = 0 : si64} : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
  %z = "onnx.Expand"(%zero, %s) : (tensor<i64>, tensor<3xi64>) -> tensor<?x5x7xi64>
  %e = "onnx.Add"(%arg0, %z) : (tensor<?x1x7xi64>, tensor<?x5x7xi64>) -> tensor<?x5x7xi64>
  %r = "onnx.Cast"(%e) {saturate = 1 : si64, to = f32} : (tensor<?x5x7xi64>) -> tensor<?x5x7xf32>
  onnx.Return %r : tensor<?x5x7xf32>
}
// CHECK-LABEL: func.func @test_sink_expand_below_cast

// -----

func.func @test_sink_expand_below_sub_rhs(%arg0: tensor<?x1x?xf32> {onnx.name = "x"}) -> (tensor<?x5x?xf32> {onnx.name = "y"}) {
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %c5 = onnx.Constant dense<5> : tensor<1xi64>
  %d0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x1x?xf32>) -> tensor<1xi64>
  %d2 = "onnx.Dim"(%arg0) {axis = 2 : si64} : (tensor<?x1x?xf32>) -> tensor<1xi64>
  %s = "onnx.Concat"(%d0, %c5, %d2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %z = "onnx.Expand"(%zero, %s) : (tensor<f32>, tensor<3xi64>) -> tensor<?x5x?xf32>
  %e = "onnx.Add"(%arg0, %z) : (tensor<?x1x?xf32>, tensor<?x5x?xf32>) -> tensor<?x5x?xf32>
  %r = "onnx.Sub"(%one, %e) : (tensor<f32>, tensor<?x5x?xf32>) -> tensor<?x5x?xf32>
  onnx.Return %r : tensor<?x5x?xf32>
}
// CHECK-LABEL: func.func @test_sink_expand_below_sub_rhs

// -----

func.func @test_sink_expand_below_mul_lhs(%arg0: tensor<?x1x?xf32> {onnx.name = "x"}, %arg1: tensor<1xf32> {onnx.name = "k"}) -> (tensor<?x5x?xf32> {onnx.name = "y"}) {
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %c5 = onnx.Constant dense<5> : tensor<1xi64>
  %d0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x1x?xf32>) -> tensor<1xi64>
  %d2 = "onnx.Dim"(%arg0) {axis = 2 : si64} : (tensor<?x1x?xf32>) -> tensor<1xi64>
  %s = "onnx.Concat"(%d0, %c5, %d2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %z = "onnx.Expand"(%zero, %s) : (tensor<f32>, tensor<3xi64>) -> tensor<?x5x?xf32>
  %e = "onnx.Add"(%arg0, %z) : (tensor<?x1x?xf32>, tensor<?x5x?xf32>) -> tensor<?x5x?xf32>
  %r = "onnx.Mul"(%e, %arg1) : (tensor<?x5x?xf32>, tensor<1xf32>) -> tensor<?x5x?xf32>
  onnx.Return %r : tensor<?x5x?xf32>
}
// CHECK-LABEL: func.func @test_sink_expand_below_mul_lhs

// -----

func.func @test_sink_expand_below_less_or_equal(%arg0: tensor<?x1x?xi64> {onnx.name = "x"}) -> (tensor<?x5x?xi1> {onnx.name = "y"}) {
  %zero = onnx.Constant dense<0> : tensor<i64>
  %k = onnx.Constant dense<5> : tensor<i64>
  %c5 = onnx.Constant dense<5> : tensor<1xi64>
  %d0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x1x?xi64>) -> tensor<1xi64>
  %d2 = "onnx.Dim"(%arg0) {axis = 2 : si64} : (tensor<?x1x?xi64>) -> tensor<1xi64>
  %s = "onnx.Concat"(%d0, %c5, %d2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %z = "onnx.Expand"(%zero, %s) : (tensor<i64>, tensor<3xi64>) -> tensor<?x5x?xi64>
  %e = "onnx.Add"(%arg0, %z) : (tensor<?x1x?xi64>, tensor<?x5x?xi64>) -> tensor<?x5x?xi64>
  %r = "onnx.LessOrEqual"(%e, %k) : (tensor<?x5x?xi64>, tensor<i64>) -> tensor<?x5x?xi1>
  onnx.Return %r : tensor<?x5x?xi1>
}
// CHECK-LABEL: func.func @test_sink_expand_below_less_or_equal

// -----

func.func @test_sink_expand_attention_mask(%arg0: tensor<?x?xi64> {onnx.name = "mask"}) -> (tensor<?x1x?x?xf32> {onnx.name = "bias"}) {
  %zero = onnx.Constant dense<0> : tensor<i64>
  %a1 = onnx.Constant dense<1> : tensor<1xi64>
  %a2 = onnx.Constant dense<2> : tensor<1xi64>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %u1 = "onnx.Unsqueeze"(%arg0, %a1) : (tensor<?x?xi64>, tensor<1xi64>) -> tensor<?x1x?xi64>
  %u2 = "onnx.Unsqueeze"(%u1, %a2) : (tensor<?x1x?xi64>, tensor<1xi64>) -> tensor<?x1x1x?xi64>
  %d0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x?xi64>) -> tensor<1xi64>
  %d1 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<?x?xi64>) -> tensor<1xi64>
  %s = "onnx.Concat"(%d0, %a1, %d1, %d1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %z = "onnx.Expand"(%zero, %s) : (tensor<i64>, tensor<4xi64>) -> tensor<?x1x?x?xi64>
  %e = "onnx.Add"(%u2, %z) : (tensor<?x1x1x?xi64>, tensor<?x1x?x?xi64>) -> tensor<?x1x?x?xi64>
  %f = "onnx.Cast"(%e) {saturate = 1 : si64, to = f32} : (tensor<?x1x?x?xi64>) -> tensor<?x1x?x?xf32>
  %inv = "onnx.Sub"(%one, %f) : (tensor<f32>, tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  %c = "onnx.Cast"(%inv) {saturate = 1 : si64, to = i1} : (tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xi1>
  %r = "onnx.Where"(%c, %min, %inv) : (tensor<?x1x?x?xi1>, tensor<f32>, tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  onnx.Return %r : tensor<?x1x?x?xf32>
}
// CHECK-LABEL: func.func @test_sink_expand_attention_mask

// -----

// Negative case: the equivalent sunk form, Expand(Cast(x)) per output.
func.func @test_not_sink_shared_expand_small_growth(%arg0: tensor<4x1x8xi64> {onnx.name = "x"}) -> (tensor<4x2x8xf32> {onnx.name = "y"}, tensor<4x2x8xi32> {onnx.name = "z"}) {
  %s = onnx.Constant dense<[4, 2, 8]> : tensor<3xi64>
  %c0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f32} : (tensor<4x1x8xi64>) -> tensor<4x1x8xf32>
  %c1 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = i32} : (tensor<4x1x8xi64>) -> tensor<4x1x8xi32>
  %r0 = "onnx.Expand"(%c0, %s) : (tensor<4x1x8xf32>, tensor<3xi64>) -> tensor<4x2x8xf32>
  %r1 = "onnx.Expand"(%c1, %s) : (tensor<4x1x8xi32>, tensor<3xi64>) -> tensor<4x2x8xi32>
  onnx.Return %r0, %r1 : tensor<4x2x8xf32>, tensor<4x2x8xi32>
}
// CHECK-LABEL: func.func @test_not_sink_shared_expand_small_growth

// -----

// Negative case: Add's own broadcast does the Expand.
func.func @test_not_sink_expand_non_scalar_operand(%arg0: tensor<?x1x?xf32> {onnx.name = "x"}, %arg1: tensor<?x5x?xf32> {onnx.name = "y"}) -> (tensor<?x5x?xf32> {onnx.name = "z"}) {
  %r = "onnx.Add"(%arg0, %arg1) : (tensor<?x1x?xf32>, tensor<?x5x?xf32>) -> tensor<?x5x?xf32>
  onnx.Return %r : tensor<?x5x?xf32>
}
// CHECK-LABEL: func.func @test_not_sink_expand_non_scalar_operand

// -----

// Negative case: Where's own broadcast does the Expands.
func.func @test_not_sink_expand_where_different_shapes(%arg0: tensor<?x1x7xi1> {onnx.name = "c"}, %arg1: tensor<?x1x7xf32> {onnx.name = "a"}, %arg2: tensor<?x5x1xf32> {onnx.name = "b"}) -> (tensor<?x5x7xf32> {onnx.name = "z"}) {
  %r = "onnx.Where"(%arg0, %arg1, %arg2) : (tensor<?x1x7xi1>, tensor<?x1x7xf32>, tensor<?x5x1xf32>) -> tensor<?x5x7xf32>
  onnx.Return %r : tensor<?x5x7xf32>
}
// CHECK-LABEL: func.func @test_not_sink_expand_where_different_shapes

// -----

// The folded result, written directly.
func.func @test_sink_expand_two_constants_still_fold(%arg0: tensor<?xi64> {onnx.name = "x"}) -> (tensor<2x?xf32> {onnx.name = "y"}) {
  %c2 = onnx.Constant dense<2> : tensor<1xi64>
  %d0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
  %s = "onnx.Concat"(%c2, %d0) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %k6 = onnx.Constant dense<6.000000e+00> : tensor<f32>
  %r = "onnx.Expand"(%k6, %s) : (tensor<f32>, tensor<2xi64>) -> tensor<2x?xf32>
  onnx.Return %r : tensor<2x?xf32>
}
// CHECK-LABEL: func.func @test_sink_expand_two_constants_still_fold

// -----

func.func @test_sink_expand_two_expands_same_shape(%arg0: tensor<?x1x?xf32> {onnx.name = "x"}, %arg1: tensor<?x1x?xf32> {onnx.name = "y"}) -> (tensor<?x5x?xf32> {onnx.name = "z"}) {
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %c5 = onnx.Constant dense<5> : tensor<1xi64>
  %d0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x1x?xf32>) -> tensor<1xi64>
  %d2 = "onnx.Dim"(%arg0) {axis = 2 : si64} : (tensor<?x1x?xf32>) -> tensor<1xi64>
  %s = "onnx.Concat"(%d0, %c5, %d2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %z = "onnx.Expand"(%zero, %s) : (tensor<f32>, tensor<3xi64>) -> tensor<?x5x?xf32>
  %ex = "onnx.Add"(%arg0, %z) : (tensor<?x1x?xf32>, tensor<?x5x?xf32>) -> tensor<?x5x?xf32>
  %ey = "onnx.Add"(%arg1, %z) : (tensor<?x1x?xf32>, tensor<?x5x?xf32>) -> tensor<?x5x?xf32>
  %r = "onnx.Add"(%ex, %ey) : (tensor<?x5x?xf32>, tensor<?x5x?xf32>) -> tensor<?x5x?xf32>
  onnx.Return %r : tensor<?x5x?xf32>
}
// CHECK-LABEL: func.func @test_sink_expand_two_expands_same_shape
