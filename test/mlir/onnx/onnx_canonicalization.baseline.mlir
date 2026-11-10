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
